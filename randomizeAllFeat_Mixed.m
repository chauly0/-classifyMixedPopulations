function patientsMixed = randomizeAllFeat_Mixed(poolSens, poolRes, numCells, numPatients, percentSens, percentRes)


%% Make computationally generated patients with mixed percentages of sensitive and resistant cells

% Initialize
iNormal = zeros(numCells*percentSens,numPatients); 
iRes = zeros(numCells*percentRes,numPatients); 

% Initialize patient array
 cellsNormal = cell((numCells*percentSens*13), numPatients);
 cellsRes = cell((numCells*percentRes*13), numPatients);
 patientsMixed = cell((numCells*13)+1, numPatients);

for kHetero = 1:numPatients

% Sample __% normal cells -- randomize the indexing 
iNormal(:,kHetero) = randperm(size(poolSens,1),numCells*percentSens)';

% Sample __% resistant cells -- randomize the indexing
iRes(:,kHetero) = randperm(size(poolRes,1),numCells*percentRes)'; 

    %% For each cell within a patient sample
    for k2Normal = 1:size(iNormal,1)  
        
        if k2Normal == 1
        cellsNormal(1:13,kHetero) = poolSens(iNormal(k2Normal,kHetero),:)';
        else        
        cellsNormal(((k2Normal-1)*13+1):(k2Normal*13),kHetero) = poolSens(iNormal(k2Normal,kHetero),:)';
        end
    end
    
    for k2Res = 1:size(iRes,1)
        
        if k2Res == 1
        cellsRes(1:13, kHetero) = poolRes(iRes(k2Res,kHetero),:)';
        else
        cellsRes(((k2Res-1)*13+1):(k2Res*13),kHetero) = poolRes(iRes(k2Res,kHetero),:)';
        end 
    end 
    

end

% Combine & label bottom of cell array with patient classification
patientsMixed = vertcat(cellsNormal, cellsRes, cellstr(repelem("Resistant",numPatients)));

    
end

    
    
    
   


