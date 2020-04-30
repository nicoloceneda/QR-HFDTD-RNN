
% -------------------------------------------------------------------------------
% 0. PREPARE THE DATA
% -------------------------------------------------------------------------------


% Import the train, validation and test sets

Y_train = table2array(readtable("NASDAQ100/data_100/Y_train.csv"));
Y_valid = table2array(readtable("NASDAQ100/data_100/Y_validate.csv"));
Y_test = table2array(readtable("NASDAQ100/data_100/Y_test.csv"));


% -------------------------------------------------------------------------------
% 1. GARCH MODEL - NORMAL
% -------------------------------------------------------------------------------


% Hyperparameter selection

losses = zeros(4,4);
minimum_loss = 1e8;

for p = 1:4
    
    for q = 1:4
        
        model_garch = garch('GARCHLags', p, 'ARCHLags', q, 'Distribution', 'Gaussian');
        estimated_garch = estimate(model_garch, Y_train, 'Display', 'off');
        cond_variance_garch = infer(estimated_garch, [Y_train; Y_valid]);
        valid_variance_garch = cond_variance_garch(length(Y_train) + 1: length(Y_train) + length(Y_valid));
        valid_sigma_garch = sqrt(valid_variance_garch);                  
        
        tau = [0.01, 0.05: 0.05: 0.95, 0.99];                            
        Q = qn_calculator(valid_sigma_garch, tau);                        
        losses(p,q) = pinball_loss_function(Y_valid, Q, tau);           
        
        if losses(p,q) < minimum_loss
            
            minimum_loss = losses(p,q);
            best_p = p;
            best_q = q;
            
        end
        
    end
    
end


% Calculate the volatility for the test set

model_garch = garch('GARCHLags', best_p, 'ARCHLags', best_q, 'Distribution', 'Gaussian');
estimated_garch = estimate(model_garch, Y_train, 'Display', 'off');
cond_variance_garch = infer(estimated_garch, [Y_train; Y_valid; Y_test]);
test_variance_garch = cond_variance_garch(length(Y_train) + length(Y_valid) + 1: length(Y_train) + length(Y_valid) + length(Y_test));
test_sigma_garch = sqrt(test_variance_garch);


% Calculate the loss for the tau set

tau = [0.01, 0.05: 0.05: 0.95, 0.99];
Q = qn_calculator(test_sigma_garch, tau);
loss = pinball_loss_function(Y_test, Q, tau);

disp("GARCH-N: Test set params and loss for tau:");
disp([best_p, best_q, loss]);


% Calculate the loss for the new tau set

tau = [0.01,0.05,0.1];
Q = qn_calculator(test_sigma_garch, tau);
loss = pinball_loss_function(Y_test, Q, tau);

disp("GARCH-N: Test set params and loss for new tau:");
disp([best_p, best_q, loss]);


% -------------------------------------------------------------------------------
% 2. GARCH MODEL - STUDENT
% -------------------------------------------------------------------------------


% Hyperparameter selection

losses = zeros(4,4);
minimum_loss = 1e8;

for p = 1:4
    
    for q = 1:4
        
        model_garch = garch('GARCHLags', p, 'ARCHLags', q, 'Distribution', 't');
        estimated_garch = estimate(model_garch, Y_train, 'Display', 'off');
        cond_variance_garch = infer(estimated_garch, [Y_train; Y_valid]);
        valid_variance_garch = cond_variance_garch(length(Y_train) + 1: length(Y_train) + length(Y_valid));
        nu = estimated_garch.Distribution.DoF;
        valid_sigma_garch = sqrt(valid_variance_garch * (nu - 2) / nu);                  
        
        tau = [0.01, 0.05: 0.05: 0.95, 0.99];                            
        Q = qt_calculator(valid_sigma_garch, tau, nu);                        
        losses(p,q) = pinball_loss_function(Y_valid, Q, tau);           
        
        if losses(p,q) < minimum_loss
            
            minimum_loss = losses(p,q);
            best_p = p;
            best_q = q;
            
        end
        
    end
    
end


% Calculate the volatility for the test set

model_garch = garch('GARCHLags', best_p, 'ARCHLags', best_q, 'Distribution', 't');
estimated_garch = estimate(model_garch, Y_train, 'Display', 'off');
cond_variance_garch = infer(estimated_garch, [Y_train; Y_valid; Y_test]);
test_variance_garch = cond_variance_garch(length(Y_train) + length(Y_valid) + 1: length(Y_train) + length(Y_valid) + length(Y_test));
nu = estimated_garch.Distribution.DoF;
test_sigma_garch = sqrt(test_variance_garch * (nu - 2) / nu); 


% Calculate the loss for the tau set

tau = [0.01, 0.05: 0.05: 0.95, 0.99];
Q = qt_calculator(test_sigma_garch, tau, nu);
loss = pinball_loss_function(Y_test, Q, tau);

disp("GARCH-t: Test set params and loss for tau:");
disp([best_p, best_q, loss]);


% Calculate the loss for the new tau set

tau = [0.01,0.05,0.1];
Q = qt_calculator(test_sigma_garch, tau, nu);
loss = pinball_loss_function(Y_test, Q, tau);

disp("GARCH-t: Test set params and loss for new tau:");
disp([best_p, best_q, loss]);


% -------------------------------------------------------------------------------
% 3. GARCH MODEL - STUDENT
% -------------------------------------------------------------------------------




% -------------------------------------------------------------------------------
% 4. FUNCTIONS
% -------------------------------------------------------------------------------


% Define a function to calculate Q for a normal distribution

function Q = qn_calculator(Y_predicted, tau)

    z_tau = norminv(tau);
    Q = Y_predicted * z_tau;
    
end


% Define a function to calculate Q for a student distribution

function Q = qt_calculator(Y_predicted, tau, nu)

    z_tau = tinv(tau, nu);
    Q = Y_predicted * z_tau;
    
end


% Define the pinball loss Function

function L = pinball_loss_function(Y_actual, Q, tau)

    error = Y_actual - Q;                               
    error_1 = tau .* error;                             
    error_2 = (tau - 1) .* error;                      
    L = mean(max(error_1, error_2), 'all');             
    
end

