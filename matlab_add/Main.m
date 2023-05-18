function Main()
%modify meta path and channel info before running
    meta='../yeast/meta.csv';
    file=fopen(meta);
    meta=textscan(file,'%d%s%d%s%s%s%s%d','Delimiter',',','HeaderLines',1);
    fclose(file);
    [n,one]=size(meta{1});

    for i=1:n
        %read table
        if meta{8}(i)==0
            fprintf('skip\n');
            continue;
        end
        idx=meta{1}(i);
        rawPath=cell2mat(meta{2}(i));
        fnum=meta{3}(i);
        maskPath=cell2mat(meta{4}(i));
        outPath=cell2mat(meta{5}(i));

        %begin processing
        AutoSegmentWithMask(rawPath,maskPath,fnum);
        
        improc2.processImageObjects(dirPathOrAnArrayCollection=rawPath);
	    improc2.launchThresholdGUI(rawPath);
        pause;
        
        if strcmp(cell2mat(meta{7}(i)),'True')
	        spotLocationSaver('tmr', rawPath,[outPath '_tmr.txt']);
        end
        if strcmp(cell2mat(meta{8}(i)),'True')
	        spotLocationSaver('cy', rawPath,[outPath '_cy.txt']);
        end
    end
end

function AutoSegmentWithMask(Path,maskPath,fnum) 
    fnumStr = sprintf('%03d',fnum);
    mask = readNPY(maskPath);
    [height, width] = size(mask);
    depth=max(mask,[],'all');
    fprintf('%d cells will be processed\n',depth);
    objects=[];
    for i = 1 : depth
        mask_local = reshape(mask==i,height, width);
        newObj = improc2.buildImageObject(mask_local, fnumStr, Path);
        objects=[objects newObj];
    end

    save(sprintf('%sdata%s.mat',Path,fnumStr),'objects');
end

function spotLocationSaver(channel, rawPath, savename)
    location = [];
    tools = improc2.launchImageObjectTools(rawPath);
    tools.iterator.goToFirstObject();
    
    while tools.iterator.continueIteration
        objectHandle = tools.objectHandle;
        bbox = objectHandle.getBoundingBox;
        results = objectHandle.getData([channel ':Spots']);
        [x,y,z] = results.getSpotCoordinates;
        x = x + bbox(2);
        y = y + bbox(1);
        arrayNum=tools.iterator.getArrayNum();
        objNum=tools.iterator.getObjNum();

        currentlocation = [x,y,z,repmat(objNum,[size(x,1) 1])]; 
        location = cat(1, location, currentlocation);
        tools.iterator.goToNextObject();
        
    end
    
    writematrix(location, savename);
end

