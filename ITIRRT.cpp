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
https://github.com/ompl/ompl/blob/1aed811a0f23023246c576338aab1d45d2ebc517/src/ompl/geometric/planners/rrt/src/RRTConnect.cpp


Author: Thibault Barbie
*/

#include <cmath>
#include <vector>
#include <map>
#include <limits>

#include "ompl/geometric/planners/rrt/ITIRRT.h"
#include "ompl/base/goals/GoalSampleableRegion.h"
#include "ompl/tools/config/SelfConfig.h"
#include "ompl/util/String.h"

#include <torch/torch.h>
#include <torch/script.h>

// We declare the module as a global variable because the python bindings do not work
torch::jit::script::Module torch_model;

ompl::geometric::ITIRRT::ITIRRT(const base::SpaceInformationPtr &si, std::string method, bool addIntermediateStates)
  : base::Planner(si, addIntermediateStates ? "ITIRRTIntermediate" : "ITIRRT")
{
    specs_.recognizedGoal = base::GOAL_SAMPLEABLE_REGION;
    specs_.directed = true;

    Planner::declareParam<double>("range", this, &ITIRRT::setRange, &ITIRRT::getRange, "0.:1.:10000.");
    Planner::declareParam<bool>("intermediate_states", this, &ITIRRT::setIntermediateStates, &ITIRRT::getIntermediateStates, "0,1");
    Planner::declareParam<std::string>("initialization_method", this, &ITIRRT::setInitializationMethod, &ITIRRT::getInitializationMethod, "forward");

    connectionPoint_ = std::make_pair<base::State *, base::State *>(nullptr, nullptr);
    distanceBetweenTrees_ = std::numeric_limits<double>::infinity();
    addIntermediateStates_ = addIntermediateStates;
    initialization_method = method;
}

void ompl::geometric::ITIRRT::setup()
{
    Planner::setup();
    tools::SelfConfig sc(si_, getName());
    sc.configurePlannerRange(maxDistance_);
    
    if (!start_tree) {start_tree.reset(tools::SelfConfig::getDefaultNearestNeighbors<Motion*>(this));}
    if (!goal_tree)  {goal_tree.reset( tools::SelfConfig::getDefaultNearestNeighbors<Motion*>(this));}
    
    start_tree->setDistanceFunction([this](const Motion* a, const Motion* b) { return distanceFunction(a, b); });
    goal_tree->setDistanceFunction( [this](const Motion* a, const Motion* b) { return distanceFunction(a, b); });
}


void ompl::geometric::ITIRRT::clear() {
    Planner::clear();
    sampler_.reset();

    std::vector<Motion* > motions;

    if (start_tree) {
      start_tree->list(motions);
      for (auto &motion : motions) {
	if (motion->state != nullptr) {si_->freeState(motion->state);}
	delete motion;
      }
    }

    if (goal_tree) {
      goal_tree->list(motions);
      for (auto &motion : motions) {
	if (motion->state != nullptr) {si_->freeState(motion->state);}
	delete motion;
      }
    }
    
    if (start_tree) {start_tree->clear();}
    if (goal_tree)  {goal_tree->clear();}
    connectionPoint_ = std::make_pair<base::State *, base::State *>(nullptr, nullptr);
    distanceBetweenTrees_ = std::numeric_limits<double>::infinity();

    initial_trajectory.clear();
}

bool ompl::geometric::ITIRRT::independent_trees_initialization(TreeData &start_tree, TreeData &goal_tree, std::vector<TreeData>& independent_trees) {
  bool solved {false};

  std::vector<Motion*> start_tree_nodes;
  std::vector<Motion*> goal_tree_nodes;
  start_tree->list(start_tree_nodes);
  goal_tree->list(goal_tree_nodes);

  Motion* start_motion { new Motion{start_tree_nodes[0]->state, nullptr, 0} };
  Motion* last_motion {start_motion};
  
  bool collided {false};
  bool is_start_tree {true};
  int tree_idx {0};
  
  for (base::State* q: initial_trajectory) {
    collision_check_count += 1;
    if (not(collided) and si_->checkMotion(last_motion->state, q)) {
      base::State* state {si_->allocState()};
      si_->copyState(state, q);    
      Motion* motion { new Motion{state, last_motion, tree_idx} }; 
      (is_start_tree ? start_tree : independent_trees.back())->add(motion);

      last_motion = motion;
    }
    // If we collide we change the tree
    else {
      collided = true; 
      // If the state is valid we create a tree from it
      if (si_->isValid(q)) {
	tree_idx += is_start_tree ? 2 : 1; // the goal tree index is 1 so we need to skip it
	  
	base::State* state {si_->allocState()};
	si_->copyState(state, q);
	Motion* motion { new Motion{state, nullptr, tree_idx} };
	last_motion = motion;
	  
	TreeData independent_tree;
	independent_tree.reset( tools::SelfConfig::getDefaultNearestNeighbors<Motion*>(this));
	independent_tree->setDistanceFunction([this](const Motion* a, const Motion* b) { return distanceFunction(a, b); });
	
	independent_tree->add(motion);
	independent_trees.push_back(independent_tree);
	
	collided = false;
	is_start_tree = false;
      }
    }  
  }

  // We try to connect the last node to the goal
  collision_check_count += 1;
  if (si_->checkMotion(last_motion->state, goal_tree_nodes[0]->state)) {
    if (is_start_tree) { 
      start_tree->list(start_tree_nodes);
      auto path(std::make_shared<PathGeometric>(si_));
      path->getStates().reserve(start_tree_nodes.size() + 1);
      
      for (Motion* m: start_tree_nodes) {path->append(m->state);}
      path->append(goal_tree_nodes.back()->state);
	
      pdef_->addSolutionPath(path, false, 0.0, getName());
      solved = true;
    }
    else {
      std::vector<Motion*> last_independent_tree_motions;
      independent_trees.back()->list(last_independent_tree_motions);
  
      // Inverse the parents and change the tree index to goal
      for (size_t i=0; i<last_independent_tree_motions.size()-1; i++) {
	last_independent_tree_motions[i]->parent = last_independent_tree_motions[i+1];
	last_independent_tree_motions[i]->tree_index = 1;
      }
      last_independent_tree_motions.back()->parent = goal_tree_nodes[0];
      last_independent_tree_motions.back()->tree_index = 1;
      
      goal_tree->add(last_independent_tree_motions);
      independent_trees.pop_back();
    }
  }

  // Remove small independent_trees
  for (int i = independent_trees.size()-1; i>=0; i--) {
    if(independent_trees[i]->size()<10) {independent_trees.erase(independent_trees.begin() + i);}
  }

  // We clear the vector to avoid increase with the next call of solve
  initial_trajectory.clear();
  
  return solved;
}

ompl::geometric::ITIRRT::TreeExtensionResult ompl::geometric::ITIRRT::extend_tree(TreeData &tree, base::State* state, Motion*& extension_motion, bool tree_turn) {
  bool reach = true;
  
  Motion* motion = new Motion{state, nullptr, 0};
  Motion* near_motion = tree->nearest(motion); 
  base::State *dstate = si_->allocState();
  si_->copyState(dstate, state);
    
  double d = si_->distance(near_motion->state, state);
  
  if (d > maxDistance_) {
    base::State* interpolated_state = si_->allocState();
    si_->getStateSpace()->interpolate(near_motion->state, state, maxDistance_ / d, interpolated_state);

    /* Check if we have moved at all. Due to some stranger state spaces (e.g., the constrained state spaces),
     * interpolate can fail and no progress is made. Without this check, the algorithm gets stuck in a loop as it
     * thinks it is making progress, when none is actually occurring. */
    if (si_->equalStates(near_motion->state, interpolated_state)) {return TRAPPED;}

    si_->copyState(dstate, interpolated_state);    
    reach = false;
  }

  
  collision_check_count += 1;
  bool validMotion = tree_turn ? si_->checkMotion(near_motion->state, dstate): si_->isValid(dstate) && si_->checkMotion(dstate, near_motion->state);

  if (!validMotion) {return TRAPPED;}
  
  if (addIntermediateStates_) {
    const base::State *astate = tree_turn ? near_motion->state : dstate;
    const base::State *bstate = tree_turn ? dstate : near_motion->state;
    
    std::vector<base::State *> states;
    const unsigned int count = si_->getStateSpace()->validSegmentCount(astate, bstate);

    if (si_->getMotionStates(astate, bstate, states, count, true, true)) {si_->freeState(states[0]);}

    for (std::size_t i = 1; i < states.size(); ++i) {
      Motion* motion = new Motion{si_->allocState(), nullptr, 0};
      motion->state = states[i];
      motion->parent = near_motion;
      motion->tree_index = near_motion->tree_index;
      tree->add(motion);

      near_motion = motion;
    }

    extension_motion = near_motion;
  }
  else {
    Motion* motion = new Motion{si_->allocState(), nullptr, 0};
    si_->copyState(motion->state, dstate);
    motion->parent = near_motion;
    motion->tree_index = near_motion->tree_index;
    tree->add(motion);

    extension_motion = motion;
  }

  return reach ? REACHED : ADVANCED;
}

ompl::base::PlannerStatus ompl::geometric::ITIRRT::solve(const base::PlannerTerminationCondition &ptc) {
  checkValidity();
  ompl::base::StateSpacePtr state_space = si_->getStateSpace();
  maxDistance_ = si_->getStateValidityCheckingResolution();

  auto *goal = dynamic_cast<base::GoalSampleableRegion *>(pdef_->getGoal().get());

  if (goal == nullptr) {
    OMPL_ERROR("%s: Unknown type of goal", getName().c_str());
    return base::PlannerStatus::UNRECOGNIZED_GOAL_TYPE;
  }

  std::vector<double> start_values;
  while (const base::State *st = pis_.nextStart()) {
    Motion* motion = new Motion{si_->allocState(), nullptr, 0};
    si_->copyState(motion->state, st);
    state_space->copyToReals (start_values, st);
    start_tree->add(motion);
  }

  if (start_tree->size() == 0) {
    OMPL_ERROR("%s: Motion planning start tree could not be initialized!", getName().c_str());
    return base::PlannerStatus::INVALID_START;
  }
  
  if (!goal->couldSample()) {
    OMPL_ERROR("%s: Insufficient states in sampleable goal region", getName().c_str());
    return base::PlannerStatus::INVALID_GOAL;
  }

  std::vector<double> goal_values;  
  if (goal_tree->size() == 0 ) {
    const base::State *st = goal_tree->size() == 0 ? pis_.nextGoal(ptc) : pis_.nextGoal();
    if (st != nullptr) {
      Motion* motion = new Motion{si_->allocState(), nullptr, 1};
      si_->copyState(motion->state, st);
      state_space->copyToReals (goal_values, st);
      goal_tree->add(motion);
    }

    if (goal_tree->size() == 0) {
      OMPL_ERROR("%s: Unable to sample any valid states for goal tree", getName().c_str());
      return base::PlannerStatus::INVALID_GOAL;
    }
  }


  // Initial trajectory generation
  generate_initial_trajectory(start_values, goal_values);

  // Trees initialization
  std::vector<TreeData> independent_trees;
  independent_trees.reserve(20);

  bool solved {independent_trees_initialization(start_tree, goal_tree, independent_trees)};

  if (!sampler_) {sampler_ = si_->allocStateSampler();}
  
  OMPL_INFORM("%s: Starting planning with %d states already in datastructure", getName().c_str(),(int)(start_tree->size() + goal_tree->size()));
  
  Motion* approxsol {nullptr};
  double approxdif {std::numeric_limits<double>::infinity()};
  Motion* random_motion {new Motion{si_->allocState(), nullptr, 0}};
  bool tree_turn {false};
  Motion* extension_motion = new Motion{si_->allocState(), nullptr, 0};

  std::vector<double> values;
  
  while (!ptc && !solved) {
    iteration_count +=1;
    
    // Change current tree
    tree_turn = !tree_turn;
    TreeData &tree = tree_turn ? start_tree : goal_tree;
    TreeData &other_tree = tree_turn ? goal_tree : start_tree;
    
    sampler_->sampleUniform(random_motion->state);

    TreeExtensionResult tree_extension_result = extend_tree(tree, random_motion->state, extension_motion, tree_turn);
    
    if (tree_extension_result != TRAPPED) {     
      base::State* added_state = si_->allocState();
      si_->copyState(added_state, extension_motion->state);
      Motion* added_motion = new Motion{added_state, extension_motion->parent, extension_motion->tree_index};
     
      // First we extend the other tree (start or goal)
      TreeExtensionResult other_tree_connect_result = ADVANCED; 
      while (other_tree_connect_result == ADVANCED) {other_tree_connect_result = extend_tree(other_tree, added_state, extension_motion, tree_turn);}
      
      // Then we extend the closest independent tree
      if(independent_trees.size()>0) {
	int min_distance_tree_index {0};
	double min_distance {std::numeric_limits<double>::max()}; 
	Motion* closest_motion = new Motion {nullptr, nullptr, 0};
	
	for(size_t i=0; i<independent_trees.size(); i++) {
	  Motion* near_motion = independent_trees[i]->nearest(added_motion);
	  double d {distanceFunction(near_motion, added_motion)};
	  if (d < min_distance) {
	    min_distance = d;
	    min_distance_tree_index = i;
	    closest_motion = near_motion;
	  } 
	}
	
	TreeExtensionResult independent_tree_connect_result = ADVANCED;
	while (independent_tree_connect_result == ADVANCED) {independent_tree_connect_result = extend_tree(tree, closest_motion->state, extension_motion, tree_turn);}
	
	// If we connect to the independent tree we merge it to the current (start or goal) tree
	if (independent_tree_connect_result == REACHED) {
	  // Because the independent tree is a simple line of state added in order
	  // we can simply iterate through the motions
	  std::vector<Motion*> independent_tree_nodes;
	  independent_trees[min_distance_tree_index]->list(independent_tree_nodes);

	  bool before_connecting_node {true};
	  int connecting_node_index {-1};
	  for(size_t i=0; i<independent_tree_nodes.size(); i++) {
	    independent_tree_nodes[i]->tree_index = added_motion->tree_index;
	  
	    if (before_connecting_node) {
	      if (si_->equalStates(independent_tree_nodes[i]->state, closest_motion->state)) {
		before_connecting_node = false;
		if (i < independent_tree_nodes.size()-1 ) {independent_tree_nodes[i+1]->parent = extension_motion;}
		if (i > 0 )                               {independent_tree_nodes[i-1]->parent = extension_motion;}
		connecting_node_index = i;
	      }
	      else { independent_tree_nodes[i]->parent = independent_tree_nodes[i+1];}
	    }
	  }
	  independent_tree_nodes.erase(independent_tree_nodes.begin() + connecting_node_index);
	  tree->add(independent_tree_nodes);
	  
	  independent_trees[min_distance_tree_index]->clear();
	  independent_trees.erase(independent_trees.begin() + min_distance_tree_index);
	}
      }
      
      const double newDist = tree->getDistanceFunction()(added_motion, other_tree->nearest(added_motion));
      if (newDist < distanceBetweenTrees_) {distanceBetweenTrees_ = newDist;}
      
      Motion* startMotion = tree_turn ? extension_motion : added_motion;
      Motion* goalMotion = tree_turn ? added_motion : extension_motion;
      
      // if we connected the trees in a valid way (start and goal pair is valid)
      if (other_tree_connect_result == REACHED) {
	// it must be the case that either the start tree or the goal tree has made some progress
	// so one of the parents is not nullptr. We go one step 'back' to avoid having a duplicate state
	// on the solution path
	if   (startMotion->parent != nullptr) {startMotion = startMotion->parent;}
	else                                  {goalMotion  = goalMotion->parent;}
	
	// Construct the solution path 
	Motion* solution = startMotion;
	std::vector<Motion*> mpath1;
	while (solution != nullptr) {
	  mpath1.push_back(solution);
	  solution = solution->parent;
	}
	
	solution = goalMotion;
	std::vector<Motion*> mpath2;
	while (solution != nullptr) {
	  mpath2.push_back(solution);
	  solution = solution->parent;
	}
	
	auto path(std::make_shared<PathGeometric>(si_));
	path->getStates().reserve(mpath1.size() + mpath2.size());
	for (int i = mpath1.size() - 1; i >= 0; --i) {path->append(mpath1[i]->state);}
	for (auto &i : mpath2) {path->append(i->state);}
	
	pdef_->addSolutionPath(path, false, 0.0, getName());
	
	solved = true;
	break;
      }
    }
  }

  // Clearing the memory
  std::vector<Motion*> motions;
  for (TreeData tree: independent_trees) {
    if (tree) {
      tree->list(motions);
      for (auto &motion : motions) {
	if (motion->state != nullptr) {si_->freeState(motion->state);}
	delete motion;
      }
    }
    if (tree) {tree->clear();}
  }
    
  OMPL_INFORM("%s: Created %u states (%u start + %u goal)", getName().c_str(), start_tree->size() + goal_tree->size(), start_tree->size(), goal_tree->size());
  
  if (approxsol && !solved) {    
    // construct the solution path 
    std::vector<Motion*> mpath;
    while (approxsol != nullptr) {
      mpath.push_back(approxsol);
      approxsol = approxsol->parent;
    }
    
    auto path(std::make_shared<PathGeometric>(si_));
    for (int i = mpath.size() - 1; i >= 0; --i) {path->append(mpath[i]->state);}
    pdef_->addSolutionPath(path, true, approxdif, getName());
    
    return base::PlannerStatus::APPROXIMATE_SOLUTION;
  }

  clear();
  return solved ? base::PlannerStatus::EXACT_SOLUTION : base::PlannerStatus::TIMEOUT;
}

void ompl::geometric::ITIRRT::load_torch_model_from_path(std::string path) {torch_model = torch::jit::load(path);}

void ompl::geometric::ITIRRT::generate_initial_trajectory(std::vector<double> start_values, std::vector<double> goal_values) {
  ompl::base::StateSpacePtr state_space = si_->getStateSpace();
  size_t state_dim = si_->getStateDimension();

  if (initialization_method == "linear") {
    // We try to divide the [0, 1] interval into multiple points
    int interpolate_point_number = 30;

    for (int i=1; i<interpolate_point_number+1; i++) {
      double t = double(i)/(interpolate_point_number+1);
      
      std::vector<double> state_coordinates (state_dim, 0);
      for(size_t j=0; j<state_dim; j++) {state_coordinates[j] = start_values[j]*(1-t)+goal_values[j]*t;}
 
      base::State* state  {si_->allocState()};
      state_space->copyFromReals(state, state_coordinates);
      
      initial_trajectory.push_back(state);
    }
    
  }
  else {
    torch::Tensor input_tensor = torch::empty({1, problem_specifications.size+start_values.size()*2});
    auto input_accessor = input_tensor.accessor<float,2>();

    for(size_t i=0; i < problem_specifications.data.size() ; i++) {input_accessor[0][i] = problem_specifications.data[i];}
    for(size_t i=0; i < start_values.size() ; i++) {input_accessor[0][problem_specifications.data.size()+i] = start_values[i];}
    for(size_t i=0; i < goal_values.size() ; i++)  {input_accessor[0][problem_specifications.data.size()+start_values.size()+i] = goal_values[i];}
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);
    
    at::Tensor output = torch_model.forward(inputs).toTensor();

    auto output_accessor = output.accessor<float,2>();

    for (size_t i=0; i<output.sizes()[1]/state_dim; i++) {
      std::vector<double> state_coordinates (state_dim, 0); // Initialize the vector to 0
      for(size_t j=0; j<state_dim; j++) {state_coordinates[j] = (double) output_accessor[0][state_dim*i+j];}
      base::State* state  {si_->allocState()};
      state_space->copyFromReals (state, state_coordinates);
      initial_trajectory.push_back(state);
    }
  }
}


void ompl::geometric::ITIRRT::getPlannerData(base::PlannerData &data) const
{
    Planner::getPlannerData(data);

    std::vector<Motion*> motions;
    if (start_tree) {start_tree->list(motions); }

    for (auto &motion : motions) {
      if (motion->parent == nullptr) {data.addStartVertex(base::PlannerDataVertex(motion->state, 1));}
      else                           {data.addEdge(base::PlannerDataVertex(motion->parent->state, 1), base::PlannerDataVertex(motion->state, 1));}
    }

    motions.clear();
    if (goal_tree) {goal_tree->list(motions);}

    for (auto &motion : motions) {
      if (motion->parent == nullptr) {data.addGoalVertex(base::PlannerDataVertex(motion->state, 2));}
      // The edges in the goal tree are reversed to be consistent with start tree
      else                           {data.addEdge(base::PlannerDataVertex(motion->state, 2), base::PlannerDataVertex(motion->parent->state, 2));}
    }

    // Add the edge connecting the two trees
    data.addEdge(data.vertexIndex(connectionPoint_.first), data.vertexIndex(connectionPoint_.second));

    // Add some info.
    data.properties["approx goal distance REAL"] = ompl::toString(distanceBetweenTrees_);
}
