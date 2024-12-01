clear all
clear functions

%Parameter_Define;
SNR = 0:3:15;
N_SNR = length(SNR);

%速度为0 3 6 9 12 15，收集他们x position
%                                   
Follow=[143.07, 144.6, 157.12, 195.48, 277.05, 314.45];
Lead=[143.08, 148.92, 167.04, 213.97, 306.21, 346.02];

Space = Lead-Follow;
%% plot & figure
figure('name','Vehicles Position'),
valuesLead = spcrv([[SNR(1) SNR SNR(end)];[Lead(1) Lead Lead(end)]],3);
valuesSpace = spcrv([[SNR(1) SNR SNR(end)];[Follow(1) Follow Follow(end)]],3);

plot(valuesLead(1,:),valuesLead(2,:),'-.b');
hold on,
plot(valuesSpace(1,:),valuesSpace(2,:),'-r');

legend(' Leading Vehicle', ' Following Vehicle');
grid on,
figure_FontSize=16;

set(findobj('FontSize',16),'FontSize',figure_FontSize);
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2.5);
set(gca,'XTick',[0:1:15]);
set(gca, 'XLim',[0, 15]);
set(gca,'YTick',[120,170,220,270,320,370]);
set(gca, 'YLim',[120,370]);
xlabel('Vehicle Speed (m/s)');
ylabel('Vehicle Position (m)');

%% plot & figure
figure('name','Vehicles Spacing'),
valuesSpace = spcrv([[SNR(1) SNR SNR(end)];[Space(1) Space Space(end)]],3);

plot(valuesSpace(1,:),valuesSpace(2,:),'-r');

legend(' Vehicle Spacing (S^*)');
grid on,
figure_FontSize=16;

set(findobj('FontSize',16),'FontSize',figure_FontSize);
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2.5);
set(gca,'XTick',[0:1:15]);
set(gca, 'XLim',[0, 15]);
set(gca, 'YLim',[0,35]);
xlabel('Vehicle Speed (m/s)');
ylabel('Vehicle Position (m)');