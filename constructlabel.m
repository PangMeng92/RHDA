function labelvector = constructlabel(ClassNumber, Sample)
% construct: function constructlabel aims to construct labelvector for face
% dataset

%          Input:
%             ClassNumber    -The number of categories
%             Sample         -The number of trainning sample or testing
%                             sample of each person

%          Output:
%             labelvector    - Vector which reflects label of face dataset
%                              
vector = [1:1:ClassNumber];
vector = vector';
matrix = repmat(vector, 1, Sample);
matrix = matrix';
totalnumber = ClassNumber * Sample;
labelvector = reshape(matrix, 1, totalnumber);