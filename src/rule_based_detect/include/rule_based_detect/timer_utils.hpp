#pragma once

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>

using time_util_t = std::chrono::time_point<std::chrono::high_resolution_clock>;
using std::cout;
using std::endl;

class TimeChecker
{
public:
  TimeChecker() {}
  TimeChecker(bool mode)
  {
    DEBUG_MODE = mode;
  }
  ~TimeChecker() {}

  void start(std::string id)
  {
    if (DEBUG_MODE)
    {
      for (const auto &target : startTime)
      {
        if (target.first == id)
        {
          cout << "already exist id" << endl;
          return;
        }
      }
    }

    std::pair<std::string, time_util_t> input;
    input.first = id;
    input.second = std::chrono::high_resolution_clock::now();
    startTime.push_back(input);
  }

  void finish(std::string id)
  {
    bool errFlag = true;
    for (const auto &target : startTime)
    {
      if (target.first == id)
      {
        errFlag = false;
        std::pair<std::string, double> input;
        input.first = id;
        input.second = (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - target.second)).count() * 1e-6;
        calcTime.push_back(input);
        break;
      }
    }
    if (errFlag)
    {
      cout << "there are no exist id" << endl;
    }
  }

  void clear()
  {
    startTime.clear();
    calcTime.clear();
  }

  void print()
  {
    size_t max_length = 0;
    for (const auto &temp : calcTime)
    {
      if (max_length < temp.first.length())
      {
        max_length = temp.first.length();
      }
    }
    std::cout << "----- Time Taken -----" << std::endl;
    for (const auto &temp : calcTime)
    {
      std::cout << std::setw(max_length + 10) << temp.first << " : " << std::fixed
                << std::setprecision(3) << std::right << temp.second << "ms" << std::endl;
    }
    std::cout << "----------------------" << std::endl;
  }

private:
  std::vector<std::pair<std::string, time_util_t>> startTime;
  std::vector<std::pair<std::string, double>> calcTime;
  bool DEBUG_MODE = true;
};