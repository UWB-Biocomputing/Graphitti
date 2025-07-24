#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// Helper to trim whitespace
std::string trim(const std::string &s)
{
   size_t start = s.find_first_not_of(" \t\r\n");
   size_t end = s.find_last_not_of(" \t\r\n");
   return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

// Parse matrices from XML file
std::map<std::string, std::vector<std::string>> parse_matrices(const std::string &filename)
{
   std::ifstream file(filename);
   std::map<std::string, std::vector<std::string>> matrices;
   std::string line, current_name;
   bool in_matrix = false;
   std::vector<std::string> values;

   while (std::getline(file, line)) {
      line = trim(line);
      if (line.find("<Matrix") == 0) {
         size_t name_pos = line.find("name=\"");
         if (name_pos != std::string::npos) {
            size_t start = name_pos + 6;
            size_t end = line.find("\"", start);
            current_name = line.substr(start, end - start);
            in_matrix = true;
            values.clear();
         }
      } else if (in_matrix && line.find("</Matrix>") == 0) {
         matrices[current_name] = values;
         in_matrix = false;
      } else if (in_matrix) {
         std::istringstream iss(line);
         std::string val;
         while (iss >> val) {
            values.push_back(val);
         }
      }
   }
   return matrices;
}

int main(int argc, char *argv[])
{
   if (argc != 3) {
      std::cerr << "Usage: compare_matrices good_output.xml test_output.xml\n";
      return 1;
   }

   auto good = parse_matrices(argv[1]);
   auto test = parse_matrices(argv[2]);

   std::set<std::string> all_vars;
   for (const auto &kv : good)
      all_vars.insert(kv.first);
   for (const auto &kv : test)
      all_vars.insert(kv.first);

   int compared = 0;
   std::vector<std::string> mismatched_vars;

   for (const auto &var : all_vars) {
      int in_good = good.count(var);
      int in_test = test.count(var);

      if (in_good == 0) {
         std::cout << "Variable '" << var
                   << "' present in test output but missing from good output.\n";
         continue;
      }
      if (in_test == 0) {
         std::cout << "Variable '" << var
                   << "' present in good output but missing from test output.\n";
         continue;
      }

      compared++;
      const auto &good_vals = good[var];
      const auto &test_vals = test[var];
      if (good_vals.size() != test_vals.size()) {
         std::cout << "Variable '" << var
                   << "' has different number of values (good: " << good_vals.size()
                   << ", test: " << test_vals.size() << ").\n";
         mismatched_vars.push_back(var);
         continue;
      }
      bool mismatch = false;
      for (size_t i = 0; i < good_vals.size(); ++i) {
         if (good_vals[i] != test_vals[i]) {
            mismatch = true;
            break;
         }
      }
      if (mismatch) {
         std::cout << "Variable '" << var << "' has mismatched values.\n";
         mismatched_vars.push_back(var);
      }
   }

   std::cout << compared << " variables compared.\n";
   if (!mismatched_vars.empty()) {
      std::cout << "Mismatched values found for variables:\n";
      for (const auto &var : mismatched_vars) {
         std::cout << "  " << var << "\n";
      }
   } else {
      std::cout << "All compared variables matched.\n";
   }
   return 0;
}