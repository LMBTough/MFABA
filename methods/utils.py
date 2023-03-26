import time
import torch
import numpy as np
import numpy as np
import torch
from scipy.integrate import trapezoid
from torchvision.transforms import GaussianBlur

def get_time(fun_name):
    def warpper(fun):
        def inner(*arg, **kwarg):
            s_time = time.time()
            res = fun(*arg, **kwarg)
            e_time = time.time()
            # print('{} ：{} FPS'.format(fun_name, math.floor(1/(e_time - s_time)* 100) / 100))            return res
            print(f"{fun_name}: {e_time - s_time} s")
            return res
        return inner
    return warpper


class VisionSensitivityN():

    def __init__(self, classifier, input_size, n, num_masks=100):
        self.classifier = classifier
        self.n = n
        self.device = next(self.classifier.parameters()).device
        self.indices, self.masks = self._generate_random_masks(
            num_masks, input_size, device=self.device)

    def evaluate(  # noqa
            self,
            heatmap: torch.Tensor,
            input_tensor: torch.Tensor,
            target: int,
            calculate_corr=False) -> dict:
        pertubated_inputs = []
        sum_attributions = []
        for mask in self.masks:
            # perturb is done by interpolation
            pertubated_inputs.append(input_tensor * (1 - mask))
            sum_attributions.append((heatmap * mask).sum())
        sum_attributions = torch.stack(sum_attributions)
        input_inputs = pertubated_inputs + [input_tensor]
        with torch.no_grad():
            input_inputs = torch.stack(input_inputs).to(self.device)
            output = self.classifier(input_inputs)
        output_pertubated = output[:-1]
        output_clean = output[-1:]

        diff = output_clean[:, target] - output_pertubated[:, target]
        score_diffs = diff.cpu().numpy()
        sum_attributions = sum_attributions.cpu().numpy()

        # calculate correlation for single image if requested
        corrcoef = None
        if calculate_corr:
            corrcoef = np.corrcoef(sum_attributions.flatten(),
                                   score_diffs.flatten())
        return {
            "correlation": corrcoef,
            "score_diffs": score_diffs,
            "sum_attributions": sum_attributions
        }

    def _generate_random_masks(self, num_masks, input_size, device='cuda:0'):
        """
        generate random masks with n pixel set to zero
        Args:
            num_masks: number of masks
            n: number of perturbed pixels
        Returns:
            masks
        """
        indices = []
        masks = []
        h, w = input_size
        for _ in range(num_masks):
            idxs = np.unravel_index(
                np.random.choice(h * w, self.n, replace=False), (h, w))
            indices.append(idxs)
            mask = np.zeros((h, w))
            mask[idxs] = 1
            masks.append(torch.from_numpy(mask).to(torch.float32).to(device))
        return indices, masks

    

import numpy as np
import torch


class GridView:
    """ access something by 2D-tile indices """

    def __init__(self, orig_dim: tuple, tile_dim: tuple):
        self.orig_r = orig_dim[0]
        self.orig_c = orig_dim[1]
        self.tile_h = tile_dim[0]
        self.tile_w = tile_dim[1]
        self.tiles_r = self.orig_r // self.tile_h
        self.tiles_c = self.orig_c // self.tile_w
        self.grid = (self.tiles_r, self.tiles_c)

        if self.orig_r % self.tile_h != 0 or self.orig_c % self.tile_w != 0:
            print("Warning: GridView is not sound")

    def tile_slice(self, tile_r: int, tile_c: int):
        """ get the slice that would return the tile r,c """
        assert tile_r < self.tiles_r, \
            "tile {} is out of range with max {}".format(tile_r, self.tiles_r)
        assert tile_c < self.tiles_c, \
            "tile {} is out of range with max {}".format(tile_c, self.tiles_c)

        r = tile_r * self.tile_h
        c = tile_c * self.tile_w

        # get pixel indices of tile
        if tile_r == self.tiles_r - 1:
            slice_r = slice(r, None)
        else:
            slice_r = slice(r, r + self.tile_h)

        if tile_c == self.tiles_c - 1:
            slice_c = slice(c, None)
        else:
            slice_c = slice(c, c + self.tile_w)

        return slice_r, slice_c


class Perturber:

    def perturb(self, r: int, c: int):
        """ perturb a tile or pixel """
        raise NotImplementedError

    def get_current(self) -> np.ndarray:
        """ get current img with some perturbations """
        raise NotImplementedError

    def get_idxes(self, hmap: np.ndarray, reverse=False) -> list:
        # TODO: might not needed, we determine perturb priority outside
        #  perturber
        """return a sorted list with shape length NUM_CELLS of
        which pixel/cell index to blur first"""
        raise NotImplementedError

    def get_grid_shape(self) -> tuple:
        """ return the shape of the grid, i.e. the max r, c values """
        raise NotImplementedError


class PixelPerturber(Perturber):

    def __init__(self, inp: torch.Tensor, baseline: torch.Tensor):
        self.current = inp.clone()
        self.baseline = baseline

    def perturb(self, r: int, c: int):
        self.current[:, r, c] = self.baseline[:, r, c]

    def get_current(self) -> torch.Tensor:
        return self.current

    def get_grid_shape(self) -> tuple:
        return self.current.shape


class GridPerturber(Perturber):

    def __init__(self, inp: torch.Tensor, baseline: torch.Tensor, tile_dim):
        assert len(tile_dim) == 2
        self.view = GridView(tuple(inp.shape[-2:]), tile_dim)
        self.current = inp.clone()
        self.baseline = baseline

    def perturb(self, r: int, c: int):
        slc = self.view.tile_slice(r, c)
        self.current[:, slc[0], slc[1]] = self.baseline[:, slc[0], slc[1]]

    def get_current(self) -> torch.Tensor:
        return self.current

    def get_grid_shape(self) -> tuple:
        return self.view.tiles_r, self.view.tiles_c

    def get_tile_shape(self) -> tuple:
        return self.view.tile_h, self.view.tile_w

class VisionInsertionDeletion():

    def __init__(self, classifier, pixel_batch_size=10, sigma=5.):
        self.classifier = classifier
        self.classifier.eval()
        self.pixel_batch_size = pixel_batch_size
        self.gaussian_blurr = GaussianBlur(int(2 * sigma - 1), sigma)

    @torch.no_grad()
    def evaluate(self, heatmap, input_tensor, target):  # noqa
        """# TODO to add docs
        Args:
            heatmap (Tensor): heatmap with shape (H, W) or (3, H, W).
            input_tensor (Tensor): image with shape (3, H, W).
            target (int): class index of the image.
        Returns:
            dict[str, Union[Tensor, np.array, float]]: a dictionary
                containing following fields:
                - del_scores: ndarray,
                - ins_scores:
                - del_input:
                - ins_input:
                - ins_auc:
                - del_auc:
        """

        # sort pixel in attribution
        num_pixels = torch.numel(heatmap)
        _, indices = torch.topk(heatmap.flatten(), num_pixels)
        indices = np.unravel_index(indices.cpu().numpy(), heatmap.size())

        # apply deletion game
        deletion_perturber = PixelPerturber(input_tensor,
                                            torch.zeros_like(input_tensor))
        deletion_scores = self._procedure_perturb(deletion_perturber,
                                                  num_pixels, indices, target)

        # apply insertion game
        blurred_input = self.gaussian_blurr(input_tensor)
        insertion_perturber = PixelPerturber(blurred_input, input_tensor)
        insertion_scores = self._procedure_perturb(insertion_perturber,
                                                   num_pixels, indices, target)

        # calculate AUC
        insertion_auc = trapezoid(
            insertion_scores, dx=1. / len(insertion_scores))
        deletion_auc = trapezoid(deletion_scores, dx=1. / len(deletion_scores))

        # deletion_input and insertion_input are final results, they are
        # only used for debug purpose
        # TODO check if it is necessary to convert the Tensors to np.ndarray
        return {
            "del_scores": deletion_scores,
            "ins_scores": insertion_scores,
            "del_input": deletion_perturber.get_current(),
            "ins_input": insertion_perturber.get_current(),
            "ins_auc": insertion_auc,
            "del_auc": deletion_auc
        }

    def _procedure_perturb(self, perturber, num_pixels, indices, target):
        """ # TODO to add docs
        Args:
            perturber (PixelPerturber):
            num_pixels (int):
            indices (tuple):
            target (int):
        Returns:
            np.ndarray:
        """
        scores_after_perturb = []
        replaced_pixels = 0
        while replaced_pixels < num_pixels:
            perturbed_inputs = []
            for i in range(80):
                batch = min(num_pixels - replaced_pixels,
                            self.pixel_batch_size)

                # perturb # of pixel_batch_size pixels
                for pixel in range(batch):
                    perturb_index = (indices[0][replaced_pixels + pixel],
                                     indices[1][replaced_pixels + pixel])

                    # perturb input using given pixels
                    perturber.perturb(perturb_index[0], perturb_index[1])
                perturbed_inputs.append(perturber.get_current())
                replaced_pixels += batch
                if replaced_pixels == num_pixels:
                    break

            # get score after perturb
            device = next(self.classifier.parameters()).device
            perturbed_inputs = torch.stack(perturbed_inputs)
            logits = self.classifier(perturbed_inputs.to(device))
            score_after = torch.softmax(logits, dim=-1)[:, target]
            scores_after_perturb = np.concatenate(
                (scores_after_perturb, score_after.detach().cpu().numpy()))
        return scores_after_perturb