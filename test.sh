n=100
declare -i sum
for ((i = 1; i <= n; i++)); do
    temp=$(python test.py)
    echo "TEMP $temp"
    sum+=$(($temp + 0))
    echo "SUM $sum"
done
sum=$(($sum/$n/1000))
echo "AVERAGE ACC: $sum"
