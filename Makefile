
default: build clean
	@echo "Done."

build:
	@echo "Building 'improc' ..."
	/usr/bin/env python cuda_slic/setup.py build_ext -i | grep error

clean:
	@echo "Cleaning ..."
	find cuda_slic/ \( -name _*.c -o -name _*.cpp \) -exec rm -rf {} \; 2> /dev/null
	rm -rf build 2> /dev/null