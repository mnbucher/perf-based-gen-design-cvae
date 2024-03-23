% http://matlab.izmiran.ru/help/techdoc/matlab_external/ch07cl19.html

clear all
clc

NQk=[40:5:100]; %kN/m
t=[11.5,17.5,24,32.4,36.5];
b=[100:25:250];
hS=[2:.25:3.5];
fk=[2.7,3.1,3.5,3.9,4.6,5.3]; %M2.5 + HLzA/B/E; FK6-20

dataset=[];
zaehler=1;

for ii=1:1 %length(NQk)
    for jj=1:length(t)
        for kk=1:length(b)
            for mm=1:length(hS)
                for nn=1:length(fk)
                    
%                     1. kopiere Masterfile für neue Instanz
%                     eval('copyfile("mw_ec6_master.xls",sprintf("Instance_%d.xls",zaehler),''f'')')   
%                     
%                     2. Fülle Features X in das Instanz-File
%                     excelapp = actxserver('Excel.Application');
%                     wkbk = excelapp.Workbooks;
%                     
%                     wdata = wkbk.Open(strcat(pwd,'\',eval('sprintf("Instance_%d.xls",zaehler)')));
%                     
%                     eActivesheetRange = get(wdata.Activesheet,'Range','C22:C22');
%                     eActivesheetRange.Value = NQk(1,ii);
%                     
%                     eActivesheetRange = get(wdata.Activesheet,'Range','C23:C23');
%                     eActivesheetRange.Value = t(1,jj);
% 
%                     eActivesheetRange = get(wdata.Activesheet,'Range','C25:C25');
%                     eActivesheetRange.Value = b(1,kk);
% 
%                     eActivesheetRange = get(wdata.Activesheet,'Range','C26:C26');
%                     eActivesheetRange.Value = hS(1,mm);
% 
%                     eActivesheetRange = get(wdata.Activesheet,'Range','C27:C27');
%                     eActivesheetRange.Value = fk(1,nn);
%                     
%                     wdata.Save; 
%                     wdata.Close; 
%                     excelapp.Quit; 
%                     excelapp.delete;
                    
                    
                    % 3. Hole Targets Y aus dem Instanz-File
                    excelapp = actxserver('Excel.Application');
                    wkbk = excelapp.Workbooks;
                    
                    wdata = wkbk.Open(strcat(pwd,'\',eval('sprintf("Instance_%d.xls",zaehler)')));                    
                   
                    eActivesheetRange = get(wdata.Sheets.Item('1'),'Range','F46:F46');
                    eta_W = eActivesheetRange.value;

                    eActivesheetRange2 = get(wdata.Sheets.Item('1'),'Range','C40:C40');
                    Phi1 = eActivesheetRange2.value;
                    
                    eActivesheetRange3 = get(wdata.Sheets.Item('1'),'Range','F40:F40');
                    Phi2 = eActivesheetRange3.value;

                    eActivesheetRange4 = get(wdata.Sheets.Item('1'),'Range','B49:B49');
                    Schlankheit = eActivesheetRange4.value;
                    switch Schlankheit
                        case "OK"
                            Schlankheit=1;
                        otherwise
                            Schlankheit=0;
                    end
                    
                    dataset(zaehler,:)=[NQk(1,ii),t(1,jj),b(1,kk),hS(1,mm),fk(1,nn),eta_W,Phi1,Phi2,Schlankheit];
                    
                    wdata.Close; 
                    excelapp.Quit; 
                    excelapp.delete;
                    
                    clear eta_W Phi1 Phi2 Schlankheit
                    zaehler=zaehler+1;
                end
            end
        end
    end
end

save dataset_1.txt dataset -ascii
save dataset_1.mat dataset

dataset = array2table(dataset,'VariableNames',{'NQk','t','b','hS','fk','eta_W','Phi1','Phi2','Schlankheit'});
parquetwrite('dataset_1.parparquet',dataset,'Version','1.0')

% Beende alle Excel Instanzen
% 1. Click Start>Run> type “cmd” and hit enter or click ‘OK’.
% 2. In the command prompt window that appears, type the following (without quotes) and hit Enter:
% taskkill /f /im excel.exe