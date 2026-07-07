```
mkdir build
cd build
cmake .. -DPY_VERSION=3.10 -DWITH_GPU=ON -DWITH_TESTING=ON -DCMAKE_BUILD_TYPE=Release -DWITH_DISTRIBUTE=ON -DWITH_PSCORE=OFF -DWITH_MKL=OFF -DCUDA_ARCH_NAME=Auto -DON_INFER=OFF
make -j

-D等价于set(CACHE)
首次配置后，该值会写入 CMakeCache.txt，后续即使删除 -D 参数也会保留
```

```
作为“容器”被修改时（如 set VAR、list ... VAR）→ 不加 ${}
作为“值”被读取时（如 message(${VAR})、${VAR}/path）→ 加 ${}
if() 中一律不加 ${}（除非特殊场景如字符串拼接）
```

# a

```
add_executable()
add_library(caffe2::cuda INTERFACE IMPORTED)
set_property(
    TARGET caffe2::cuda PROPERTY INTERFACE_LINK_LIBRARIES
    CUDA::cuda_driver)
```

```
add_subdirectory(${SUBDIR})
将其注册为 CMake 子项目，CMake 会自动执行该子目录中的构建逻辑。
```

# f

```
file(GLOB SUBDIRECTORIES 
     LIST_DIRECTORIES true 
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
     ${CMAKE_CURRENT_SOURCE_DIR}/*
)
file(模式匹配 output arg1 value1 arg2 valu2 pattern)
```

```
find_package
```

ctest​​ 是 ​​CMake​​ 的测试工具

/paddle/Paddle/test/CMakeLists.txt

```
py_test_modules(test_fused_gemm_epilogue_rule MODULES
                    test_fused_gemm_epilogue_rule)
                    
bash_test_modules(
    test_comm_group_num
    START_BASH
    test_comm_group_num.sh
    
//py_test_modules是函数名，TARGET_NAME是第一个参数
function(py_test_modules TARGET_NAME)
  if(WITH_TESTING)
      set(multiValueArgs MODULES DEPS ENVS)
      //设置multiValueArgs的值，传递多个参数（自动加分号）
      cmake_parse_arguments(py_test_modules "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})
      //必须按顺序写
      //cmake_parse_arguments(prefix ... ... MODULES;DEPS;ENVS 实参)
//解析后的参数会以 <prefix>_<arg> 格式保存。此处前缀是 py_test_modules，因此 MODULES 对应的变量名为 py_test_modules_MODULES
//py_test_modules_MODULES = "a;b;c"
//add_test 是 CMake 自带的命令
      add_test(
        NAME ${TARGET_NAME}
//若 ${TARGET_NAME} 是 test_reduction_rule，则运行测试时可通过 ctest -R test_reduction_rule 单独执行。
        COMMAND
          ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
          ${py_test_modules_ENVS} ${FLAGS_PIR_MODE} ${PYTHON_EXECUTABLE}
          ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

function(bash_test_modules TARGET_NAME)
  set(oneValueArgs TIMEOUT START_BASH)
  set(multiValueArgs DEPS ENVS LABELS)
  cmake_parse_arguments(bash_test_modules "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
    add_test(
      NAME ${TARGET_NAME}
      COMMAND
        ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
        TEST_TARGET_NAME=${TARGET_NAME} TEST_TIMEOUT=${timeout}
        ${bash_test_modules_ENVS} bash
        ${CMAKE_CURRENT_BINARY_DIR}/${bash_test_modules_START_BASH}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
```

```
/paddle/Paddle/tools/test_runner.py

for module_name in sys.argv[1:]:
    test_loader = unittest.TestLoader()
    module = importlib.import_module(module_name)
    tests = test_loader.loadTestsFromModule(module)
    res = unittest.TextTestRunner(stream=buffer).run(tests)

test_fused_gemm_epilogue_rule.py

import unittest
class TestExample(unittest.TestCase):
    def test_add(self):
        ...
```

unittest 是 Python 标准库自带的测试框架，必须继承 TestCase 类，方法以test_开头

pytest 是第三方库，测试文件和函数以test_开头

# l

```
list(REMOVE_ITEM CURRENT_DIRS tests)
list(移除 input&output pattern)
```

# s

```
set(ASCEND_COMPUTE_UNIT "ascend910b" CACHE STRING "soc that need to be compiled")
option(BUILD_OPEN_PROJECT "Build open ascend ops project." ON)
缓存变量，持久化到 CMakeCache.txt

set(FOO "cache" CACHE STRING "")
set(FOO "normal")          # 普通变量遮蔽缓存变量
set(FOO "new_cache" CACHE STRING "")  # ❌ 失败！缓存变量仍为 "cache"
set(FOO "forced" CACHE STRING "" FORCE)  # ✅ 成功强制覆盖
读取 ${VAR} → 优先返回普通变量
```

# t

```
target_sources(op_host_aclnn PRIVATE ...)
向 op_host_aclnn 添加源文件
PRIVATE 表示该源文件仅用于当前目标编译，不传递给依赖此目标的其他目标
```

```
target_link_libraries(torch_global_deps ${Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS})
向 torch_global_deps 添加链接库
```



