
%% 调用拟合神经网络获取距离并导出xlsx
table_rssi = readtable("导出rssi值.xlsx", VariableNamingRule="preserve");
data_rssi = table2array(table_rssi(1:end, 2:end));
output = [];
for i = 1 : 1 : length(data_rssi)
    rssi = data_rssi(: , i);
    dist = RSSI2dist(rssi);
    output(:,i) = dist;
    disp(dist)
end
xlswrite("dist.xlsx", output)



