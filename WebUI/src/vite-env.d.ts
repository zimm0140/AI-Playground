/// <reference types="vite/client" />
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
