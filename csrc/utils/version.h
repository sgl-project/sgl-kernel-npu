// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License

#ifndef SGL_KERNEL_NPU_VERSION_H
#define SGL_KERNEL_NPU_VERSION_H

/* version information */
#define VERSION_MAJOR 0
#define VERSION_MINOR 1
#define VERSION_FIX 0

/* second level marco define 'CONCAT' to get string */
#define CONCAT(x, y, z) x.##y.##z
#define STR(x) #x
#define CONCAT2(x, y, z) CONCAT(x, y, z)
#define STR2(x) STR(x)

/* get cancat version string */
#define LIB_VERSION STR2(CONCAT2(VERSION_MAJOR, VERSION_MINOR, VERSION_FIX))

#ifndef GIT_LAST_COMMIT
#define GIT_LAST_COMMIT empty
#endif

/*
 * global lib version string with build time
 */
[[maybe_unused]] static const char *LIB_VERSION_FULL =
    "library version: " LIB_VERSION ", commit: " STR2(GIT_LAST_COMMIT);

#endif  // SGL_KERNEL_NPU_VERSION_H
