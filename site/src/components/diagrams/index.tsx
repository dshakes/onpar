// Importable diagram components. Embed in MDX with a client directive:
// import { LoopWalkthrough } from '../../../components/diagrams';
// <LoopWalkthrough client:visible />
import { FlowDiagram } from "./FlowDiagram";
import { LOOP } from "./flows-onpar";

export { FlowDiagram } from "./FlowDiagram";
export const LoopWalkthrough = () => <FlowDiagram def={LOOP} />;
