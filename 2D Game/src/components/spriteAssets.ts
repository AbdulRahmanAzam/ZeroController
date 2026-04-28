export interface Frame {
  filename: string;
  frame: {
    x: number;
    y: number;
    w: number;
    h: number;
  };
  spriteSourceSize?: {
    x: number;
    y: number;
    w: number;
    h: number;
  };
  sourceSize: {
    w: number;
    h: number;
  };
}

export interface SpriteAtlas {
  textures: Array<{
    image: string;
    size: {
      w: number;
      h: number;
    };
    frames: Frame[];
  }>;
}

export const atlasCache = new Map<string, SpriteAtlas>();
export const imageReadyCache = new Set<string>();

const SPRITE_ASSETS = [
  { atlas: '/sprites/player1/knight.json', image: '/sprites/player1/knight.png' },
] as const;

export async function preloadSprites(): Promise<void> {
  await Promise.all(
    SPRITE_ASSETS.map(async ({ atlas, image }) => {
      if (!atlasCache.has(atlas)) {
        try {
          const data = await fetch(atlas).then(r => r.json()) as SpriteAtlas;
          atlasCache.set(atlas, data);
        } catch (e) {
          console.warn('[SpriteSheet] Could not preload atlas:', atlas, e);
        }
      }

      if (!imageReadyCache.has(image)) {
        await new Promise<void>(resolve => {
          const img = new Image();
          img.onload = () => { imageReadyCache.add(image); resolve(); };
          img.onerror = () => resolve();
          img.src = image;
        });
      }
    }),
  );
}