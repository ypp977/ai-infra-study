# 使用的编译器
NVCC = nvcc

# 编译选项（-O2 优化，-arch=sm_70 可根据你的GPU架构调整）
NVCC_FLAGS = -O2 -arch=sm_70

# 源码目录（当前目录下所有 .cu）
SRC = $(wildcard *.cu)

# 输出目录
BIN_DIR = bin

# 把 .cu 文件名替换成可执行文件名，并放到 bin/ 下
BIN = $(patsubst %.cu,$(BIN_DIR)/%,$(SRC))

# 默认目标：编译所有可执行文件
all: $(BIN)

# 确保 bin 目录存在
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# 规则：.cu -> bin/可执行文件
$(BIN_DIR)/%: %.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# 清理生成的可执行文件
clean:
	rm -rf $(BIN_DIR)
