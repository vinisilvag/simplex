mkdir -p outputs

for i in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15
do
  echo "Instance $i:\n"
  python3 main.py ./instances/$i > ./outputs/$i 
  diff ./outputs/$i ./expected/$i
done
