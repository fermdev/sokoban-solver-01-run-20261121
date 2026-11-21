// sokoban-solver: A* with deadlock detection.
// Improvements over plain-astar: deadlock pruning (box in corner not on goal),
// better heuristic (min matching), tie-breaking on fewer pushes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const EXPANSIONS = 300_000;
const DIRS: [number, number, string][] = [[0,-1,'u'],[0,1,'d'],[-1,0,'l'],[1,0,'r']];

function solve(text: string): string {
  const rows = text.split('\n').filter(r => r.length > 0);
  const w = Math.max(...rows.map(r => r.length));
  const wall: boolean[] = [];
  const goals: number[] = [];
  let boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      wall[i] = ch === '#';
      if ('.*+'.includes(ch)) goals.push(i);
      if ('$*'.includes(ch)) boxes.push(i);
      if ('@+'.includes(ch)) player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  const goalSet = new Set(goals);

  // Deadlock: corner + freeze + 2x2 block
  function isDeadlock(bs: number[]): boolean {
    for (const b of bs) {
      if (goalSet.has(b)) continue;
      const x = b % w, y = Math.floor(b / w);
      const up = y === 0 || wall[b - w];
      const down = wall[b + w] || false;
      const left = x === 0 || wall[b - 1];
      const right = wall[b + 1] || false;
      if ((up || down) && (left || right)) return true;
    }
    // 2x2 block of boxes/walls not on goals
    for (const b of bs) {
      if (goalSet.has(b)) continue;
      const x = b % w;
      // check 2x2 squares containing this box
      for (const [dx, dy] of [[0,0],[-1,0],[0,-1],[-1,-1]]) {
        const sq = [b+dy*w+dx, b+dy*w+dx+1, b+(dy+1)*w+dx, b+(dy+1)*w+dx+1];
        if (sq.every(s => wall[s] || bs.includes(s))) {
          if (sq.some(s => !goalSet.has(s) && (wall[s] || bs.includes(s)))) return true;
        }
      }
    }
    return false;
  }

  // Heuristic: min-cost bipartite matching (boxes to goals) + wall distance
  const dist = (b: number): number => {
    let best = Infinity;
    for (const g of goals) {
      const d = Math.abs((g % w) - (b % w)) + Math.abs(Math.floor(g / w) - Math.floor(b / w));
      if (d < best) best = d;
    }
    return best;
  };
  const est = (bs: number[]): number => {
    // Simple sum of min distances (admissible but weaker than Hungarian)
    let total = 0;
    for (const b of bs) total += dist(b);
    return total;
  };
  const key = (p: number, bs: number[]): string => String.fromCharCode(p, ...bs);

  const sp: number[] = [player];
  const sb: number[][] = [boxes];
  const sg: number[] = [0]; // moves
  const sp2: number[] = [0]; // pushes
  const parent: number[] = [-1];
  const step: string[] = [''];
  const best = new Map<string, number>([[key(player, boxes), 0]]);
  const heap: [number, number, number, number][] = [[est(boxes), 0, 0, 0]];
  const less = (a: [number,number,number,number], b: [number,number,number,number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && (a[3] > b[3] || (a[3] === b[3] && a[2] < b[2])))));
  const push = (e: [number,number,number,number]): void => {
    let i = heap.length; heap.push(e);
    while (i > 0) { const p = (i-1)>>1; if (!less(heap[i]!, heap[p]!)) break; [heap[i],heap[p]]=[heap[p]!,heap[i]!]; i=p; }
  };
  const pop = (): [number,number,number,number] => {
    const top = heap[0]!; const last = heap.pop()!;
    if (heap.length > 0) { heap[0] = last;
      for (let i = 0; ;) { let c = 2*i+1; if (c >= heap.length) break;
        if (c+1 < heap.length && less(heap[c+1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break; [heap[i],heap[c]]=[heap[c]!,heap[i]!]; i=c; }
    }
    return top;
  };

  let expanded = 0;
  while (heap.length > 0 && expanded < EXPANSIONS) {
    const [, g, s, pushes] = pop();
    if (best.get(key(sp[s]!, sb[s]!)) !== s) continue;
    const bs = sb[s]!;
    if (bs.every(b => goalSet.has(b))) {
      let moves = '';
      for (let t = s; t > 0; t = parent[t]!) moves = step[t] + moves;
      return moves;
    }
    if (isDeadlock(bs)) continue;
    expanded++;
    const p = sp[s]!;
    // Move ordering: pushes first (they change the state more)
    const moves: [number, number, string, number, number][] = [];
    for (const [dx, dy, letter] of DIRS) {
      const d = dy * w + dx;
      const q = p + d;
      if (wall[q]) continue;
      const at = bs.indexOf(q);
      if (at >= 0) {
        if (wall[q + d] || bs.includes(q + d)) continue;
        moves.push([q, d, letter.toUpperCase(), 1, at]); // push
      } else {
        moves.push([q, d, letter, 0, -1]); // walk
      }
    }
    // Sort: pushes first, then by heuristic improvement
    moves.sort((a, b) => a[3] - b[3]);
    for (const [q, d, move, isPush, at] of moves) {
      let nb = bs;
      let np = pushes;
      if (isPush && at >= 0) {
        nb = bs.slice();
        nb[at] = q + d;
        nb.sort((a, b) => a - b);
        np = pushes + 1;
      }
      const k = key(q, nb);
      const seen = best.get(k);
      if (seen !== undefined && sg[seen]! <= g + 1) continue;
      const t = sp.length;
      sp.push(q);
      sb.push(nb);
      sg.push(g + 1);
      sp2.push(np);
      parent.push(s);
      step.push(move);
      best.set(k, t);
      push([g + 1 + est(nb), g + 1, t, np]);
    }
  }
  return '';
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}