function bvhWriteFile(fileName, skel, channels, frameLength)

% BVHWRITEFILE Write a bvh file from a given structure and channels.
% FORMAT
% DESC writes a bvh file from a given structure and channels.
% ARG fileName : the file name to use.
% ARG skel : the skeleton structure to use.
% ARG channels : the channels to use.
% ARG frameLength : the length of a frame.
%
% COPYRIGHT : Neil D. Lawrence, 2006
%
% SEEALSO : bvhReadFile
 
% MOCAP

if nargin < 4
  frameLength = 0.03333;
end

global SAVECHANNELS
SAVECHANNELS = [];

fid = fopen(fileName, 'w');
depth = 0;
printedNodes = [];
fprintf(fid, 'HIERARCHY\n');
i = 1;
while ~any(printedNodes==i)
  printedNodes = printNode(fid, 1, skel, printedNodes, depth, channels);
end
fprintf(fid, 'MOTION\n');
fprintf(fid, 'Frames: %d\n', size(channels, 1));
fprintf(fid, 'Frame Time: %.6f\n', frameLength);
for i = 1:size(channels, 1)
  for j = 1:size(channels, 2)
    fprintf(fid, '%.6f ', SAVECHANNELS(i, j));
  end
  fprintf(fid, '\n');
end
fclose(fid);



function printedNodes = printNode(fid, j, skel, printedNodes, ...
                                  depth, channels);

% PRINTNODE Print out the details from the given node.

global SAVECHANNELS

prePart = computePrePart(depth);
if depth > 0
  if strcmp(skel.tree(j).name, 'Site')
    fprintf(fid, [prePart 'End Site\n']);
  else
    fprintf(fid, [prePart 'JOINT %s\n'], skel.tree(j).name);
  end
else
  fprintf(fid, [prePart 'ROOT %s\n'], skel.tree(j).name);
end
fprintf(fid, [prePart '{\n']);
depth = depth + 1;
prePart = computePrePart(depth);
fprintf(fid, [prePart 'OFFSET %.6f %.6f %.6f\n'], ...
        skel.tree(j).offset(1), ...
        skel.tree(j).offset(2), ...
        skel.tree(j).offset(3));
ichannel = 1;
channelcount = length(skel.tree(j).channels);
if ~strcmp(skel.tree(j).name, 'Site')
    fprintf(fid, [prePart 'CHANNELS %d'], channelcount);
    while(ichannel < channelcount + 1)    
        channelstr = skel.tree(j).channels{ichannel};
        if any(strcmp('Xposition', channelstr))
            SAVECHANNELS = [SAVECHANNELS channels(:, skel.tree(j).posInd(1))];
            fprintf(fid, ' Xposition');
        elseif any(strcmp('Yposition', channelstr))
            SAVECHANNELS = [SAVECHANNELS channels(:, skel.tree(j).posInd(2))];
            fprintf(fid, ' Yposition');
        elseif any(strcmp('Zposition', channelstr))
            SAVECHANNELS = [SAVECHANNELS channels(:, skel.tree(j).posInd(3))];
            fprintf(fid, ' Zposition');
        elseif any(strcmp('Zrotation', channelstr))
            SAVECHANNELS = [SAVECHANNELS channels(:, skel.tree(j).rotInd(3))];
            fprintf(fid, ' Zrotation');
        elseif any(strcmp('Xrotation', channelstr))
            SAVECHANNELS = [SAVECHANNELS channels(:, skel.tree(j).rotInd(1))];
            fprintf(fid, ' Xrotation');
        elseif any(strcmp('Yrotation', channelstr))
            SAVECHANNELS = [SAVECHANNELS channels(:, skel.tree(j).rotInd(2))];
            fprintf(fid, ' Yrotation');
        end
        ichannel = ichannel + 1; % update the ichannel
    end
    fprintf(fid, '\n');
end

% print out channels
printedNodes = j;
for i = skel.tree(j).children
  printedNodes = [printedNodes printNode(fid, i, skel, printedNodes, ...
                                         depth, channels)];
end
depth = depth - 1;
prePart = computePrePart(depth);
fprintf(fid, [prePart '}\n']);


function prePart = computePrePart(depth);

prePart = [];
for i = 1:depth
  prePart = [prePart '\t'];
end
