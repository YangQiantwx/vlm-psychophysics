export const LEVELS={sf:[.5,1,2,4,8,16,32,64],m:[.05,.1,.2,.3,.4,.6,.7,.8,.9,1],tf:[5],duration:[3]};
export const DEFAULTS={sf:4,m:.4,tf:5,duration:3,direction:'RIGHT',phase:Math.PI/24};
export function calibrate(source,p){return {dc:.31,k:.31*p.m,mu:.31,calibrationVersion:'fixed_classic_mean_Y_v18'};}
export function phaseText(p){return (p.phase*180/Math.PI).toFixed(1)+'°';}
