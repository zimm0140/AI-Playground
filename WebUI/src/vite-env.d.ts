/// <reference types="vite/client" />
// eslint-disable-next-line @typescript-eslint/no-unused-vars
import 'vue'

declare module 'vue' {
  interface ComponentCustomProperties {
    languages: StringKV
  }
}

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<Record<string, unknown>, Record<string, unknown>, unknown>
  export default component
}
