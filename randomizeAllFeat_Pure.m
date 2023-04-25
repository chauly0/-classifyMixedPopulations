function patientsPure = randomizeAllFeat_Pure(poolSens, numCells, numPatients)


%% Generating a pure population of drug sensitive cells

    %Initialize    
    iHomog = zeros(numCells,numPatients);
    patientsPure = cell((numCells*13)+1, numPatients);
        
        for kHomog = 1:numPatients
           
        iHomog(:,kHomog) = randperm(size(poolSens,1),numCells)';
 
            for k2Homog = 1:size(iHomog,1)

            if k2Homog == 1
            patientsPure(1:13,kHomog) = poolSens(iHomog(k2Homog,kHomog),:)';
            else
            patientsPure(((k2Homog-1)*13+1):(k2Homog*13),kHomog) = poolSens(iHomog(k2Homog,kHomog),:)';   
            end
            end 
        end 

        % Label bottom of cell array with patient classification
        patientsPure(end,:) = cellstr(repelem("Sensitive",numPatients));
    
    
    
end

    
    
    
    
    
