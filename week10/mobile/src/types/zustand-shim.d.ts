declare module 'zustand/index.js' {
  export * from 'zustand';
  const create: typeof import('zustand').create;
  export default create;
}

declare module 'zustand/middleware.js' {
  export * from 'zustand/middleware';
}