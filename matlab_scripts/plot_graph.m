s=uint8(extramatch(:,2)'./8+1);
t=uint8(extramatch(:,1)'./8+1);
G = digraph(s,t);
plot(G,'Layout','layered')