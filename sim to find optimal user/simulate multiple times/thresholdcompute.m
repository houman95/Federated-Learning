% Load the Excel file
% Load the data from the Excel file
clear all
filename = '10 slots (threshold).xlsx'; % Replace with your actual filename
% Load the data from the Excel file
data = readtable(filename, 'ReadVariableNames', true);

% Check the number of columns in the table
numColumns = size(data, 2);

% Define expected column names based on your data
% Modify this list based on the actual number of columns in your dataset
expectedColumnNames = {'timeframe', 'actives', 'user1', 'user2', 'user3', 'user4', 'user5', 'user6', 'user7', 'user8', 'user9', 'user10', 'servergrad'};

% Adjust expectedColumnNames if the dataset has fewer or more columns
expectedColumnNames = expectedColumnNames(1:numColumns);

% Apply the column names
data.Properties.VariableNames = expectedColumnNames;

% Convert the 'timeframe' column to categorical or string if necessary
if iscellstr(data.timeframe) || isstring(data.timeframe)
    data.timeframe = categorical(data.timeframe);
end

% Get the unique timeframes
timeframes = unique(data.timeframe);

% Initialize a vector to store the optimized thresholds
optimized_thresholds = zeros(length(timeframes), 1);

% Loop over each unique timeframe
for i = 2:length(timeframes)
    timeframe = timeframes(i);
    
    % Extract the data for this timeframe
    timeframe_data = data(data.timeframe == timeframe, :);
    
    % Extract the active users and server gradients
    active_users_data = timeframe_data.actives;
    server_gradients = timeframe_data.servergrad;
    
    % Extract the user gradient columns
    user_columns = contains(data.Properties.VariableNames, 'user');
    user_gradients = timeframe_data{:, user_columns};
    
    % Define the objective function to minimize
    objective = @(threshold) norm(sum(user_gradients >= (threshold * server_gradients), 2) - active_users_data);
    
    % Use fminbnd to find the optimal threshold (bounded between 0 and 1)
    optimal_threshold = fminbnd(objective, 0, 1);
    
    % Store the optimal threshold
    optimized_thresholds(i) = optimal_threshold;
end

% Create a table for the results
optimized_thresholds_table = table(timeframes, optimized_thresholds, 'VariableNames', {'Timeframe', 'OptimalThreshold'});

% Display the optimized thresholds
disp(optimized_thresholds_table);

% Optionally, save the optimized thresholds to a CSV file
writetable(optimized_thresholds_table, 'optimized_thresholds.csv');
