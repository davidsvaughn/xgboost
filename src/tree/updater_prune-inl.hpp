/*!
 * Copyright 2014 by Contributors
 * \file updater_prune-inl.hpp
 * \brief prune a tree given the statistics
 * \author Tianqi Chen
 */
#ifndef XGBOOST_TREE_UPDATER_PRUNE_INL_HPP_
#define XGBOOST_TREE_UPDATER_PRUNE_INL_HPP_

#include <vector>
#include "./param.h"
#include "./updater.h"
#include "./updater_sync-inl.hpp"

namespace xgboost {
namespace tree {
/*! \brief pruner that prunes a tree after growing finishes */
class TreePruner: public IUpdater {
 public:
  virtual ~TreePruner(void) {}
  // set training parameter
  virtual void SetParam(const char *name, const char *val) {
    using namespace std;
    param.SetParam(name, val);
    syncher.SetParam(name, val);
    if (!strcmp(name, "silent")) silent = atoi(val);
  }
  // update the tree, do pruning
  virtual void Update(const std::vector<bst_gpair> &gpair,
                      IFMatrix *p_fmat,
                      const BoosterInfo &info,
                      const std::vector<RegTree*> &trees) {
    // rescale learning rate according to size of trees
    float lr = param.learning_rate;
    param.learning_rate = lr / trees.size();
    for (size_t i = 0; i < trees.size(); ++i) {
      this->DoPrune(*trees[i]);
    }
    param.learning_rate = lr;
    syncher.Update(gpair, p_fmat, info, trees);
  }

 private:
  // try to prune off current leaf
  inline int TryPruneLeaf(RegTree &tree, int nid, int depth, int npruned) { // NOLINT(*)
    if (tree[nid].is_root()) return npruned;
    int pid = tree[nid].parent();
    RegTree::NodeStat &s = tree.stat(pid);
    ++s.leaf_child_cnt;
    if (s.leaf_child_cnt >= 2 && param.need_prune(s.loss_chg, depth - 1)) {
      // need to be pruned
      tree.ChangeToLeaf(pid, param.learning_rate * s.base_weight);
      // tail recursion
      return this->TryPruneLeaf(tree, pid, depth - 1, npruned+2);
    } else {
      return npruned;
    }
  }
  /*! \brief do pruning of a tree */
  inline void DoPrune(RegTree &tree) { // NOLINT(*)
    int npruned = 0;
    // initialize auxiliary statistics
    for (int nid = 0; nid < tree.param.num_nodes; ++nid) {
      tree.stat(nid).leaf_child_cnt = 0;
    }
    for (int nid = 0; nid < tree.param.num_nodes; ++nid) {
      if (tree[nid].is_leaf()) {
        npruned = this->TryPruneLeaf(tree, nid, tree.GetDepth(nid), npruned);
      }
    }
    if (silent == 0) {
      utils::Printf("tree pruning end, %d roots, %d extra nodes, %d pruned nodes, max_depth=%d\n",
                    tree.param.num_roots, tree.num_extra_nodes(), npruned, tree.MaxDepth());
    }
  }

 private:
  // synchronizer
  TreeSyncher syncher;
  // shutup
  int silent;
  // training parameter
  TrainParam param;
};
}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_UPDATER_PRUNE_INL_HPP_
