BUILD_DIR ?= build

.PHONY: all clean test

all:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR)
	@echo ""
	@echo "  Built: libtflite_patch.so"
	@echo ""

clean:
	cmake -E rm -rf $(BUILD_DIR) libtflite_patch.so tests/libmock_tflite.so

test: all
	python3 -m unittest discover -s tests -p "test_*.py" -v

