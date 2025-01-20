#!/bin/bash

# Path to config file where all topics are listed on new lines
CONFIG_FILE="topics_to_record.txt" 

# Read the topics into an array
readarray -t TOPICS < "${CONFIG_FILE}"

OUTPUT_FILE="./bags/in_dist.bag" #specify bag name and path
DURATION=40 #specify bag duration (10: 10 seconds; 10m: 10 minutes; 10h: 10 hours)

# Construct the rosbag record command
COMMAND="rosbag record"
COMMAND+=" -O ${OUTPUT_FILE}" #add output file
COMMAND+=" --duration=${DURATION}" #add duration

#add all topics to the command
for TOPIC in "${TOPICS[@]}"; do
    COMMAND+=" ${TOPIC}"
done


echo "Executing: $COMMAND"
#record rosbag command
$COMMAND
