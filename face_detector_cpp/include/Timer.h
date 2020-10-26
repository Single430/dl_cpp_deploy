//
// Created by zbl on 2020/10/24.
//

#ifndef TIMER_H
#define TIMER_H


#include <string>
#include <stack>
#include <chrono>

class Timer{
public:
  std::stack<std::chrono::high_resolution_clock::time_point> startend_stack;

  void start() {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    startend_stack.push(t1);
  }

  double end(std::string msg="", bool flag=true) {
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startend_stack.top()).count();
    if(msg.size() > 0){
      if (flag)
        printf("%s time elapsed: %f ms\n", msg.c_str(), duration);
    }

    startend_stack.pop();
    return duration;
  }

  void reset(){
    startend_stack = std::stack<std::chrono::high_resolution_clock::time_point>();
  }
};

#endif //TIMER_H
