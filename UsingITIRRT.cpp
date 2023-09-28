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

Author: Thibault Barbie
*/

#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/rrt/ITIRRT.h>
#include <ompl/geometric/SimpleSetup.h>

#include <ompl/config.h>
#include <iostream>

float tunnel_y = 0.4;
float tunnel_size = 0.02;
  
bool isStateValid(const ompl::base::State *state)
{
  double x = state->as<ompl::base::RealVectorStateSpace::StateType>()->values[0];
  double y = state->as<ompl::base::RealVectorStateSpace::StateType>()->values[1];
  if((x<0.2 or x>0.8) or (y>tunnel_y-tunnel_size/2.0 and y<tunnel_y+tunnel_size/2.0)) {
    return true;
  }
  else { return false; }
}

int main(){
 
  auto space(std::make_shared<ompl::base::RealVectorStateSpace>(2));

  ompl::base::RealVectorBounds bounds(2);
  bounds.setLow(0);
  bounds.setHigh(1);

  space->setBounds(bounds);

  auto si(std::make_shared<ompl::base::SpaceInformation>(space));

  si->setStateValidityChecker(isStateValid);

  ompl::base::ScopedState<> start(space);
  start->as<ompl::base::RealVectorStateSpace::StateType>()->values[0] = 0.05;
  start->as<ompl::base::RealVectorStateSpace::StateType>()->values[1] = 0.05;
  
  ompl::base::ScopedState<> goal(space);
  goal->as<ompl::base::RealVectorStateSpace::StateType>()->values[0] = 0.95;
  goal->as<ompl::base::RealVectorStateSpace::StateType>()->values[1] = 0.05;

  auto pdef(std::make_shared<ompl::base::ProblemDefinition>(si));
  pdef->setStartAndGoalStates(start, goal);

  auto planner(std::make_shared<ompl::geometric::ITIRRT>(si));
  planner->load_torch_model_from_path("tunnel_model.pt");

  // If you want to use a linear initialization instead of a model
  // planner->setInitializationMethod("linear");
  
  auto problem_specifications = planner->get_problem_specifications_container();
  problem_specifications.size = 1;
  problem_specifications.data[0] = tunnel_y;
  planner->set_problem_specifications(problem_specifications);
  
  planner->setProblemDefinition(pdef);
  planner->setup();

  ompl::base::PlannerStatus solved = planner->ompl::base::Planner::solve(10.0);

  if (solved) {
    ompl::base::PathPtr path = pdef->getSolutionPath();
    std::cout << "Found solution:" << std::endl;
    path->print(std::cout);
  }
  else {std::cout << "No solution found" << std::endl;}
  
  return 0;
}
