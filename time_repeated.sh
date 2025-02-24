#!/bin/bash

get_duration () {
    local version=$1
    shift
    if [[ ${version} -eq 0 ]]; then
        local time_keyword="Global took"
    else
        local time_keyword="Shared took"
    fi
    build/master_thesis $* -v ${version} \
        | grep -i "${time_keyword}" \
        | tr -cd 0-9
}

echo_with_comma () {
    local IFS=","
    echo "$*"
}

run_count=$1
# Remove script parameters before parsing program parameters.
shift
# Assumes positional arguments before optional arguments.
dim_x=$1
dim_y=$2

./build.sh
if [ $? -ne 0 ]; then
    exit 1
fi

distribution=uniform
version=global

# Parse arguments to remove those overwritten by this script.
args=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -d)
            # Replace default with user defined value.
            distribution=$2
            shift
            shift
            ;;
        -v)
            # Replace default with user defined value.
            version=$2
            shift
            shift
            ;;
        *)
            args+=($1)
            shift
            ;;
    esac
done

echo -n "Selected particle distribution: "
case ${distribution} in
    uniform)
        echo uniform
        ;;
    pattern_2d)
        echo 2D pattern
        ;;
    *)
        echo file
        ;;
esac

durations=()
echo "Dimensions: ${dim_x}x${dim_y}"
for i in $(seq 1 ${run_count}); do
    echo -n "Iteration ${i} "
    durations+=($(get_duration ${version} ${args[*]} -d ${distribution}))
    echo "done"
done

filepath=output/durations_repeated.csv
> ${filepath}
echo_with_comma ${durations[@]} >> ${filepath}

exit 0
