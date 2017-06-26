if [ ! -d build ]; then
  echo "Creating build"
  mkdir build
fi

cd build
cmake ..
if [ $1 = "ELLC" ]; then 
  rm -rf ELLC 

else 
  echo nothing  
fi
make -j4

