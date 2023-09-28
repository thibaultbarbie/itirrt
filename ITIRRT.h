/*
Copyright (c) 2023 OMRON Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the “Software”), to deal in the
Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


This code has been implemented from the RRTConnect algorithm made
by Ioan Sucan:
https://github.com/ompl/ompl/blob/1aed811a0f23023246c576338aab1d45d2ebc517/src/ompl/geometric/planners/rrt/RRTConnect.h


Author: Thibault Barbie
*/


#pragma once

#include "ompl/datastructures/NearestNeighbors.h"
#include "ompl/geometric/planners/PlannerIncludes.h"

#include <vector>

namespace ompl
{
    namespace geometric
    {
        /** \brief RRT-Connect4 (ITIRRT) */
        class ITIRRT : public base::Planner
        {
        public:
	  ITIRRT(const base::SpaceInformationPtr &si, std::string method="FI", bool addIntermediateStates = false);
	  
	  void getPlannerData(base::PlannerData &data) const override;
	  
	  base::PlannerStatus solve(const base::PlannerTerminationCondition &ptc) override;
	  
	  void clear() override;

	  // Return true if the intermediate states generated along motions are to be added to the tree itself
	  bool getIntermediateStates() const {return addIntermediateStates_;}

          //Specify whether the intermediate states generated along motions are to be added to the tree itself 
	  void setIntermediateStates(bool addIntermediateStates) {addIntermediateStates_ = addIntermediateStates;}

	  /** \brief Set the range the planner is supposed to use.
	      
	      This parameter greatly influences the runtime of the
	      algorithm. It represents the maximum length of a
	      motion to be added in the tree of motions. */
	  void setRange(double distance) {maxDistance_ = distance;}

	  /** \brief Get the range the planner is using */
	  double getRange() const {return maxDistance_;}
	  
	  void load_torch_model_from_path(std::string path);
	  
	  void setInitializationMethod(std::string method) {initialization_method = method;}
	  std::string getInitializationMethod() const {return initialization_method;}
	  
	  struct ProblemSpecifications {
	    std::vector<float> data;
	    size_t size;
	  };
	    
	  ProblemSpecifications get_problem_specifications_container() {
	    size_t size = 5000;
	    std::vector<float> data (size, 0);
	    return ProblemSpecifications {data, size};
	  };
	  
	  void set_problem_specifications(ProblemSpecifications ps) {
	    problem_specifications = ps;
	    problem_specifications.data.resize(ps.size);
	  };
	  
	  /** \brief Set a different nearest neighbors datastructure */
	  template <template <typename T> class NN>
	  void setNearestNeighbors()
	  {
	    if ((start_tree && start_tree->size() != 0) || (goal_tree && goal_tree->size() != 0)) {OMPL_WARN("Calling setNearestNeighbors will clear all states.");}
	    clear();
	    start_tree = std::make_shared<NN<Motion *>>();
	    goal_tree = std::make_shared<NN<Motion *>>();
	    setup();
	  }

	  void setup() override;

	  // Getter and setter for iterations and collision check numbers
	  void reset_iteration_count()       {iteration_count = 0;}
	  void reset_collision_check_count() {collision_check_count = 0;}
	  int get_iteration_count()       {return iteration_count;}
	  int get_collision_check_count() {return collision_check_count;}
	  
        private:
	  /** \brief Representation of a motion */
	  struct Motion {
	    base::State* state{nullptr};
	    Motion*      parent{nullptr};
	    int      tree_index{0};
	  };

	  /** \brief A nearest-neighbor datastructure representing a tree of motions */
	  using TreeData = std::shared_ptr<NearestNeighbors<Motion*>>;
	  
	  /** \brief The state of the tree after an attempt to extend it */
	  enum TreeExtensionResult {TRAPPED, ADVANCED, REACHED};

	  bool independent_trees_initialization(TreeData &start_tree, TreeData &goal_tree, std::vector<TreeData>& independent_trees);
	  void generate_initial_trajectory(std::vector<double> start_values, std::vector<double> goal_values);
	 
	  std::string initialization_method;
	  
	  /** \brief Compute distance between motions (actually distance between contained states) */
	  double distanceFunction(const Motion* a, const Motion* b) const {return si_->distance(a->state, b->state);}
	  
	  /** \brief Grow a tree towards a random state */
	  TreeExtensionResult extend_tree(TreeData &tree, base::State* state, Motion*& extension_motion, bool tree_turn);
	  
	  /** \brief State sampler */
	  base::StateSamplerPtr sampler_;
	  
	  TreeData start_tree;
	  TreeData goal_tree;

	  /** \brief The maximum length of a motion to be added to a tree */
	  double maxDistance_{0.};

	  /** \brief Flag indicating whether intermediate states are added to the built tree of motions */
	  bool addIntermediateStates_;

	  /** \brief The random number generator */
	  RNG rng_;

	  /** \brief The pair of states in each tree connected during planning.  Used for PlannerData computation */
	  std::pair<base::State *, base::State *> connectionPoint_;

	  /** \brief Distance between the nearest pair of start tree and goal tree nodes. */
	  double distanceBetweenTrees_;

	  ProblemSpecifications problem_specifications;

	  std::vector<base::State*> initial_trajectory;

	  int iteration_count {0};
	  int collision_check_count {0};
        };
    }
}

