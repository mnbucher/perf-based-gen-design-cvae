clear all
clc

% Annahme: die Daten liegen als Instanz mit der Form x = [NQk,t,b,hS,fk,] (Zeilenvektor) vor

zaehler = 1;    % Nummer der Instanz


% 1. kopiere Masterfile für neue Instanz
eval('copyfile("mw_ec6_master.xls",sprintf("Instance_%d.xls",zaehler),''f'')')   
                    
% 2. Fülle Features X in das Instanz-File
excelapp = actxserver('Excel.Application');
wkbk = excelapp.Workbooks;
                    
wdata = wkbk.Open(strcat(pwd,'\',eval('sprintf("Instance_%d.xls",zaehler)')));
                    
eActivesheetRange = get(wdata.Activesheet,'Range','C22:C22');
eActivesheetRange.Value = x(1,1);
                    
eActivesheetRange = get(wdata.Activesheet,'Range','C23:C23');
eActivesheetRange.Value = x(1,2);

eActivesheetRange = get(wdata.Activesheet,'Range','C25:C25');
eActivesheetRange.Value = b(1,3);

eActivesheetRange = get(wdata.Activesheet,'Range','C26:C26');
eActivesheetRange.Value = hS(1,4);

eActivesheetRange = get(wdata.Activesheet,'Range','C27:C27');
eActivesheetRange.Value = fk(1,5);
                    
wdata.Save; 
wdata.Close; 
excelapp.Quit; 
excelapp.delete;
                    
% 3. Hole Targets Y aus dem Instanz-File
excelapp = actxserver('Excel.Application');
wkbk = excelapp.Workbooks;
                    
wdata = wkbk.Open(strcat(pwd,'\',eval('sprintf("Instance_%d.xls",zaehler)')));                    
                   
eActivesheetRange = get(wdata.Sheets.Item('1'),'Range','F46:F46');
eta_W = eActivesheetRange.value;
                    
evaluations_dataset(zaehler,:)=[x,eta_W];

wdata.Close; 
excelapp.Quit; 
excelapp.delete;
                    
zaehler=zaehler+1;
