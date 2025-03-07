/*
 * Intel Level Zero Device Lister
 * ------------------------------
 * This utility program lists all Intel Level Zero devices (GPUs) available on the system.
 * 
 * It outputs a JSON-formatted array containing information about each device:
 * - id: A sequential identifier starting from 0
 * - name: The device name string
 * - device_id: The Intel device ID (if available)
 * 
 * This information can be parsed by other applications to discover and select
 * appropriate Intel GPUs for acceleration tasks.
 */

#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;

int main() {
  // Start JSON array output
  std::cout << "[";
  
  // Initialize device counter
  unsigned int id = 0;
  
  // Iterate through all platforms
  for (const auto &plt : platform::get_platforms()) {
    // Skip platforms that don't use the Level Zero backend
    if (plt.get_backend() != backend::ext_oneapi_level_zero)
      continue;

    // Iterate through all devices on this platform
    for (const auto &dev : plt.get_devices()) {
      // Add comma separator between devices (except before the first one)
      if (id > 0)
        std::cout << ", ";
      
      // Get device name
      std::string name = dev.get_info<info::device::name>();
      
      // Output the base device information
      std::cout << "{\"id\": " << id << ", \"name\": \"" << name << "\"";
      
      // Add Intel device ID if available
      if (dev.has(aspect::ext_intel_device_id)) {
        int device_id = dev.get_info<ext::intel::info::device::device_id>();
        std::cout << ", \"device_id\": " << device_id;
      }
      
      // Close the JSON object
      std::cout << "}";
      
      // Increment device counter
      id++;
    }
  }
  
  // Close JSON array and output
  std::cout << "]" << std::endl;
}
