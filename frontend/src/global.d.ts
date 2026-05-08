declare module "*.png" {
  const src: string;
  export default src;
}

declare module "react" {
  export type ChangeEvent<T = Element> = {
    target: T;
  };
  export type Dispatch<A> = (value: A) => void;
  export type SetStateAction<S> = S | ((prevState: S) => S);
  export function useState<S>(initialState: S): [S, Dispatch<SetStateAction<S>>];
  export function useRef<T>(initialValue: T | null): { current: T | null };
  const React: unknown;
  export default React;
}

declare module "react-dom/client" {
  export function createRoot(container: Element | DocumentFragment): {
    render(children: unknown): void;
  };
}

declare module "react/jsx-runtime" {
  export const Fragment: unique symbol;
  export function jsx(type: unknown, props: unknown, key?: unknown): unknown;
  export function jsxs(type: unknown, props: unknown, key?: unknown): unknown;
}

declare namespace JSX {
  interface IntrinsicElements {
    [elemName: string]: unknown;
  }
}
