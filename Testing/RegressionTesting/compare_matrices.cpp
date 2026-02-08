/*
 * @file compare_matrices.cpp
 *
 * @brief This file is used to compare an output matrix against a known good matrix
 *
 * @ingroup Testing
 */

#include <charconv>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>


//Anonymous namespace for static/re-usable methods
namespace {
   // Tolerances for floating-point comparisons
   // Absolute tolerance for near-zero values
   constexpr double EPSILON = 1e-6;
   // Relative tolerance for larger values
   constexpr double RELATIVE_EPSILON = 1e-6;

   // Helper for floating-point comparison with relative tolerance
   bool approximately_equal(double a, double b)
   {
      double diff = std::abs(a - b);
      // Pass if absolute difference is small enough (handles near-zero cases)
      if (diff <= EPSILON) {
         return true;
      }
      // Otherwise, check relative tolerance for larger values
      double max_val = std::max(std::abs(a), std::abs(b));
      return diff / max_val <= RELATIVE_EPSILON;
   }

   struct MatrixData {
      std::string name;
      std::string type;
      int rows = 0;
      int columns = 0;
      double multiplier = 1.0;
      std::vector<double> values;

      static bool parse_all_fields(const std::string &line, MatrixData &current)
      {
         std::string_view line_view = line;
         size_t pos = 0;
         bool parse_success = true;

         while (pos < line_view.size()) {
            size_t eq_pos = line_view.find("=\"", pos);
            if (eq_pos == std::string_view::npos) {
               break;
            }

            // Find field name (scan backwards for space or <)
            size_t field_start = line_view.find_last_of(" <", eq_pos);
            if (field_start == std::string_view::npos) {
               break;
            }
            field_start++;

            std::string_view field = line_view.substr(field_start, eq_pos - field_start);
            size_t value_start = eq_pos + 2;
            size_t value_end = line_view.find('\"', value_start);
            if (value_end == std::string_view::npos) {
               break;
            }

            std::string_view value = line_view.substr(value_start, value_end - value_start);

            if (field == "name") {
               current.name = value;
            } else if (field == "type") {
               current.type = value;
            } else if (field == "rows") {
               auto [ptr, ec]
                  = std::from_chars(value.data(), value.data() + value.size(), current.rows);
               if (ec != std::errc()) {
                  std::cerr << "Warning: Failed to parse 'rows' field\n";
                  parse_success = false;
               }
            } else if (field == "columns") {
               auto [ptr, ec]
                  = std::from_chars(value.data(), value.data() + value.size(), current.columns);
               if (ec != std::errc()) {
                  std::cerr << "Warning: Failed to parse 'columns' field\n";
                  parse_success = false;
               }
            } else if (field == "multiplier") {
               auto [ptr, ec]
                  = std::from_chars(value.data(), value.data() + value.size(), current.multiplier);
               if (ec != std::errc()) {
                  std::cerr << "Warning: Failed to parse 'multiplier' field\n";
                  parse_success = false;
               }
            }

            pos = value_end + 1;
         }
         return parse_success;
      }
   };


   // Helper to trim whitespace
   void trim_inplace(std::string &input)
   {
      size_t start = input.find_first_not_of(" \t\r\n");
      if (start == std::string::npos) {
         input.clear();
         return;
      }
      size_t end = input.find_last_not_of(" \t\r\n");
      // Erase trailing whitespace first, then leading
      input.erase(end + 1);
      input.erase(0, start);
   }

   // Parse matrices from XML file
   // Uses simple string matching and assumes XML elements are on separate lines.
   // Since we control the XML file structure, we don't need any specialized XML parsing.
   // If the structure changes in the future, this method may need to be updated.
   std::unordered_map<std::string, MatrixData> parse_matrices(const std::string &filename)
   {
      //Check if file exists/can be opened before continuing
      std::ifstream file(filename);
      if (!file.is_open()) {
         std::cerr << "Failed to open file: " << filename << "\n";
         return {};
      }

      std::unordered_map<std::string, MatrixData> matrices;
      std::string line;
      MatrixData current;
      bool in_matrix = false;

      while (std::getline(file, line)) {
         trim_inplace(line);
         if (line.find("<Matrix") == 0) {
            current = MatrixData {};
            if (!MatrixData::parse_all_fields(line, current)) {
               std::cerr << "Warning: Failed to parse some fields for matrix '" << current.name
                         << "'\n";
            }

            // Pre-allocate space for values if we know the size
            int expected_size = current.rows * current.columns;
            if (expected_size > 0) {
               current.values.reserve(expected_size);
            }

            in_matrix = true;
         } else if (in_matrix && line.find("</Matrix>") == 0) {
            matrices[current.name] = std::move(current);
            in_matrix = false;
         } else if (in_matrix) {
            const char *ptr = line.data();
            const char *end = ptr + line.size();
            while (ptr < end) {
               while (ptr < end && (*ptr == ' ' || *ptr == '\t'))
                  ++ptr;
               if (ptr >= end)
                  break;
               double val = 0.0;
               auto [next, ec] = std::from_chars(ptr, end, val);
               if (ec == std::errc()) {
                  current.values.push_back(val);
                  ptr = next;
               } else {
                  ++ptr;
               }
            }
         }
      }
      return matrices;
   }

   void compare_matrix_data(const MatrixData &good, const MatrixData &test,
                            std::vector<std::string> &mismatches)
   {
      bool has_mismatch = false;
      bool dimensions_match = true;

      // Check metadata
      if (good.type != test.type) {
         std::cout << "Matrix '" << good.name << "': type mismatch (good: " << good.type
                   << ", test: " << test.type << ")\n";
         has_mismatch = true;
      }
      if (good.rows != test.rows) {
         std::cout << "Matrix '" << good.name << "': rows mismatch (good: " << good.rows
                   << ", test: " << test.rows << ")\n";
         has_mismatch = true;
         dimensions_match = false;
      }
      if (good.columns != test.columns) {
         std::cout << "Matrix '" << good.name << "': columns mismatch (good: " << good.columns
                   << ", test: " << test.columns << ")\n";
         has_mismatch = true;
         dimensions_match = false;
      }
      if (!approximately_equal(good.multiplier, test.multiplier)) {
         std::cout << "Matrix '" << good.name << "': multiplier mismatch (good: " << good.multiplier
                   << ", test: " << test.multiplier << ")\n";
         has_mismatch = true;
      }

      // Skip value comparison if dimensions don't match
      if (!dimensions_match) {
         mismatches.push_back(good.name);
         return;
      }

      // Check data size
      if (good.values.size() != test.values.size()) {
         std::cout << "Matrix '" << good.name
                   << "': value count mismatch (good: " << good.values.size()
                   << ", test: " << test.values.size() << ")\n";
         has_mismatch = true;
      } else {
         // Check individual values with relative tolerance
         for (size_t i = 0; i < good.values.size(); ++i) {
            if (!approximately_equal(good.values[i], test.values[i])) {
               std::cout << "Matrix '" << good.name << "': value mismatch at index " << i
                         << " (good: " << good.values[i] << ", test: " << test.values[i] << ")\n";
               has_mismatch = true;
               break;   // Only report first mismatch per matrix
            }
         }
      }

      if (has_mismatch) {
         mismatches.push_back(good.name);
      }
   }
}   // namespace

int main(int argc, char *argv[])
{
   if (argc != 3) {
      std::cerr << "Usage: compare_matrices good_output.xml test_output.xml\n";
      return 1;
   }

   auto good = parse_matrices(argv[1]);
   auto test = parse_matrices(argv[2]);

   // Check if parsing succeeded for each file independently
   if (good.empty()) {
      std::cerr << "Warning: Good file is empty or failed to parse: " << argv[1] << "\n";
   }
   if (test.empty()) {
      std::cerr << "Warning: Test file is empty or failed to parse: " << argv[2] << "\n";
   }
   if (good.empty() && test.empty()) {
      std::cerr << "Error: Both files are empty or failed to parse.\n";
      return 1;
   }

   size_t compared = 0;
   size_t missing_in_test = 0;
   size_t extra_in_test = 0;
   std::vector<std::string> mismatched_vars;

   // Iterate over good matrices, check against test
   for (const auto &[name, good_data] : good) {
      auto in_test = test.find(name);
      if (in_test == test.end()) {
         std::cout << "Matrix '" << name
                   << "' present in good output but missing from test output.\n";
         missing_in_test++;
         continue;
      }
      compared++;
      compare_matrix_data(good_data, in_test->second, mismatched_vars);
   }

   // Check for matrices in test that aren't in good
   for (const auto &[name, _] : test) {
      if (good.find(name) == good.end()) {
         std::cout << "Matrix '" << name
                   << "' present in test output but missing from good output.\n";
         extra_in_test++;
      }
   }

   std::cout << compared << " matrices compared.\n";
   if (missing_in_test > 0) {
      std::cout << missing_in_test << " matrices missing in test output.\n";
   }
   if (extra_in_test > 0) {
      std::cout << extra_in_test << " extra matrices in test output.\n";
   }
   if (!mismatched_vars.empty()) {
      std::cout << "Mismatches found in " << mismatched_vars.size() << " matrices:\n";
      for (const auto &var : mismatched_vars) {
         std::cout << "  " << var << "\n";
      }
   } else {
      std::cout << "All compared matrices matched.\n";
   }

   // Return non-zero exit code if there were any issues
   bool has_issues = !mismatched_vars.empty() || missing_in_test > 0 || extra_in_test > 0;
   return has_issues ? 1 : 0;
}