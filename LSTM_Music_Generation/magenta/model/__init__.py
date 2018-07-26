# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Pulls in all magenta libraries that are in the public API.

To regenerate this list based on the py_library dependencies of //magenta:

bazel query 'kind(py_library, deps(//magenta))' | \
  grep '//magenta' | \
  egrep  -v "/([^:/]+):\1$" | \
  sed -e 's/\/\//import /' -e 's/\//./' -e 's/:/./' -e  's/py_pb2/pb2/' | \
  LANG=C sort
"""
