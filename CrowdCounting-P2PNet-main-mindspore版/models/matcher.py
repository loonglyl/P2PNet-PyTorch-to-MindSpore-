# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import mindspore
from scipy.optimize import linear_sum_assignment
from mindspore import nn
import mindspore.ops as ops


class HungarianMatcher_Crowd(nn.Cell):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_point: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the foreground object
            cost_point: This is the relative weight of the L1 error of the points coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        assert cost_class != 0 or cost_point != 0, "all costs cant be 0"

    # @torch.no_grad()  # mindspore的正向计算默认了no_grad
    def construct(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "points": Tensor of dim [batch_size, num_queries, 2] with the predicted point coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_points] (where num_target_points is the number of ground-truth
                           objects in the target) containing the class labels
                 "points": Tensor of dim [num_target_points, 2] containing the target point coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_points)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_prob = ops.softmax(ops.flatten(outputs["pred_logits"], start_dim=0, end_dim=1), -1)
        # out_points = outputs["pred_points"].flatten(0, 1)  # [batch_size * num_queries, 2]
        out_points = ops.flatten(outputs["pred_points"], start_dim=0, end_dim=1)

        # Also concat the target labels and points
        # print([v["labels"] for v in targets])
        tgt_ids = ops.cat([v["labels"] for v in targets])
        tgt_points = ops.cat([v["point"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L2 cost between point
        # cost_point = torch.cdist(out_points, tgt_points, p=2)
        # tgt_points = mindspore.Tensor(tgt_points, dtype=mindspore.float32)
        # cost_point = ops.cdist(out_points, tgt_points, p=2.0)
        cost_point = ops.cdist(out_points, mindspore.Tensor(tgt_points, mindspore.float32), p=2.0)

        # Compute the giou cost between point

        # Final cost matrix
        C = self.cost_point * cost_point + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1)
        sizes = [len(v["point"]) for v in targets]
        indices = [linear_sum_assignment(c[i].asnumpy()) for i, c in enumerate(C.split(sizes, -1))]
        return [(mindspore.Tensor.from_numpy(i), mindspore.Tensor.from_numpy(j)) for i, j in indices]


def build_matcher_crowd(args):  # args
    return HungarianMatcher_Crowd(cost_class=args.set_cost_class, cost_point=args.set_cost_point)
