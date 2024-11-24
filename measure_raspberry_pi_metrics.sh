#!/bin/bash

# Function to get throttling status and interpret it
get_throttled_status() {
    local throttled
    throttled=$(vcgencmd get_throttled | awk -F= '{print $2}')

    case "$throttled" in
        0x0)
            echo "Throttled: No issues detected"
            ;;
        0x50000)
            echo "Throttled: Under-voltage occurred"
            ;;
        0x20000)
            echo "Throttled: Currently under-voltage"
            ;;
        *)
            echo "Throttled: Issues detected (code: $throttled)"
            ;;
    esac
}

# Loop to monitor metrics
while true; do
    # Get the temperature
    temp=$(vcgencmd measure_temp | awk -F= '{print $2}' | sed 's/\'C//')

    # Get throttling status
    throttled_status=$(get_throttled_status)

    # Print the metrics
    echo "Temperature: $temp"
    echo "$throttled_status"
    echo "----------------------------------"

    # Wait 5 seconds before the next iteration
    sleep 5
done
