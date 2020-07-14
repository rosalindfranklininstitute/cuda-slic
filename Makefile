
default: build clean
	@echo "Done."

build:
	@echo "Building 'improc' ..."
	/usr/bin/env python survos2/improc/setup.py build_ext -i | grep error

clean:
	@echo "Cleaning ..."
	find survos2/improc/ \( -name _*.c -o -name _*.cpp \) -exec rm -rf {} \; 2> /dev/null
	rm -rf build 2> /dev/null