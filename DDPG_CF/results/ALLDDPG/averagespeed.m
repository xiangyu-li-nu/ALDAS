A = 3000:500:5000;
N_A = length(A);
x=1:N_A;


Krauss = [11.25 8.94 7.45 6.25 5.39 ];
CACC = [15.58 12.77 10.58 9.17 8.05];
DDPG = [18.82 15.57 13.34 11.32 9.87];
DDPGALL = [21.5 18.85 15.68 13.54 10.82];

figure('name','Average speed against vehicle density'),
plot(A(x),Krauss(x),'-+k','MarkerSize',10);
hold on,
plot(A(x),CACC(x),'-h','MarkerSize',10,'Color',[0.9290 0.6940 0.1250]);
hold on,
plot(A(x),DDPG(x),'-vr','MarkerSize',10);
hold on,
plot(A(x),DDPGALL(x),'-*b','MarkerSize',10);

legend(' Krauss', ' CACC', ' DDPG (with human-control)', ' DDPG (all CAV control)');
grid on,
figure_FontSize=12;

set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');

set(findobj('FontSize',12),'FontSize',figure_FontSize);
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2.5);
set(gca,'XTick',[3000 3500 4000 4500 5000]);
set(gca, 'XLim',[2800, 5200]);
set(gca,'YTick',[3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22]);
set(gca, 'YLim',[3, 22]);
xlabel('Traffic Flow (veh/h)');
ylabel('Average Speed (m/s)');