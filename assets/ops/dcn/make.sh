rm -rf build
rm -rf ./*.so
python setup.py build_ext --inplace
