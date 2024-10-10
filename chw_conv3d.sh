cd conv
clang++ chw_conv3d.cpp ../utils/utils.cpp -I/usr/local/include -L/usr/local/lib -o chw_conv3d -lcnpy -lz --std=c++11 -Wl,-rpath,/usr/local/lib
./chw_conv3d
cd ../python
python conv.py
python validate.py ../outputs/chw_conv3d_cpp.bin ../outputs/py_chw_conv3d_output.bin
cd ..
