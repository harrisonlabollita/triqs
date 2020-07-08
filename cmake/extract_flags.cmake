###################################################################################
#
# TRIQS: a Toolbox for Research in Interacting Quantum Systems
#
# Copyright (C) 2019-2020 Simons Foundation
#    author: N. Wentzell
#
# TRIQS is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# TRIQS. If not, see <http://www.gnu.org/licenses/>.
#
###################################################################################

# Recursively fetch all targets that the interface of a target depends upon
macro(get_all_interface_targets name target)
  get_property(TARGET_LINK_LIBRARIES TARGET ${target} PROPERTY INTERFACE_LINK_LIBRARIES)
  foreach(lib IN LISTS TARGET_LINK_LIBRARIES)
    if(TARGET ${lib})
      # Append to list
      list(APPEND ${name}_INTERFACE_TARGETS ${lib})
      # Recure into target dependencies
      get_all_interface_targets(${name} ${lib})
    endif()
  endforeach()
endmacro()

# Extract the property from the target and recursively from all targets it depends upon
macro(get_property_recursive)
  cmake_parse_arguments(get_property_recursive "" "TARGET" "PROPERTY" ${ARGN})
  set(target ${get_property_recursive_TARGET})
  set(property ${get_property_recursive_PROPERTY})
  get_all_interface_targets(${target} ${target})
  foreach(t IN LISTS ${target}_INTERFACE_TARGETS ITEMS ${target})
    get_property(p TARGET ${t} PROPERTY ${property})
    list(APPEND ${ARGV0} ${p})
  endforeach()
  # Clean duplicates and any occurance of '/usr/include' dirs
  if(${ARGV0})
    list(REMOVE_DUPLICATES ${ARGV0})
    list(REMOVE_ITEM ${ARGV0} /usr/include)
  endif()
endmacro()

# Recursively fetch all compiler flags attached to the interface of a target
macro(extract_flags)

  cmake_parse_arguments(ARG "BUILD_INTERFACE" "" "" ${ARGN})

  set(target ${ARGV0})
  unset(${target}_CXXFLAGS)
  unset(${target}_LDFLAGS)

  get_property_recursive(opts TARGET ${target} PROPERTY INTERFACE_COMPILE_OPTIONS)
  foreach(opt ${opts})
    set(${target}_LDFLAGS "${${target}_LDFLAGS} ${opt}")
    set(${target}_CXXFLAGS "${${target}_CXXFLAGS} ${opt}")
  endforeach()

  get_property_recursive(defs TARGET ${target} PROPERTY INTERFACE_COMPILE_DEFINITIONS)
  foreach(def ${defs})
    set(${target}_CXXFLAGS "${${target}_CXXFLAGS} -D${def}")
  endforeach()

  get_property_recursive(inc_dirs TARGET ${target} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
  get_property_recursive(sys_inc_dirs TARGET ${target} PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
  if(inc_dirs)
    list(REMOVE_ITEM sys_inc_dirs ${inc_dirs})
  endif()
  foreach(dir ${inc_dirs})
    if(NOT dir STREQUAL "/usr/include")
      set(${target}_CXXFLAGS "${${target}_CXXFLAGS} -I${dir}")
    endif()
  endforeach()
  foreach(dir ${sys_inc_dirs})
    if(NOT dir STREQUAL "/usr/include")
      set(${target}_CXXFLAGS "${${target}_CXXFLAGS} -isystem${dir}")
    endif()
  endforeach()

  get_property_recursive(libs TARGET ${target} PROPERTY INTERFACE_LINK_LIBRARIES)
  foreach(lib ${libs})
    if(NOT TARGET ${lib} AND NOT IS_DIRECTORY ${lib})
      set(${target}_LDFLAGS "${${target}_LDFLAGS} ${lib}")
    endif()
  endforeach()

  # We have to replace generator expressions explicitly
  if(ARG_BUILD_INTERFACE)
    string(REGEX REPLACE "\\$<BUILD_INTERFACE:([^ ]*)>" "\\1" ${target}_LDFLAGS "${${target}_LDFLAGS}")
    string(REGEX REPLACE "\\$<BUILD_INTERFACE:([^ ]*)>" "\\1" ${target}_CXXFLAGS "${${target}_CXXFLAGS}")
  else()
    string(REGEX REPLACE "\\$<INSTALL_INTERFACE:([^ ]*)>" "\\1" ${target}_LDFLAGS "${${target}_LDFLAGS}")
    string(REGEX REPLACE "\\$<INSTALL_INTERFACE:([^ ]*)>" "\\1" ${target}_CXXFLAGS "${${target}_CXXFLAGS}")
  endif()
  string(REGEX REPLACE " [^ ]*\\$<[^ ]*:[^>]*>" "" ${target}_LDFLAGS "${${target}_LDFLAGS}")
  string(REGEX REPLACE " [^ ]*\\$<[^ ]*:[^>]*>" "" ${target}_CXXFLAGS "${${target}_CXXFLAGS}")
endmacro()
