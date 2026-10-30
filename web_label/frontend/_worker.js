import routes from './asset-routes.json';

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    let path;
    try { path = decodeURIComponent(url.pathname); } catch { return new Response('Bad URL', {status:400}); }
    const key = Object.prototype.hasOwnProperty.call(routes,path) ? routes[path] : null;
    if (!key) return env.ASSETS.fetch(request);
    if (!['GET','HEAD'].includes(request.method)) return new Response('Method not allowed',{status:405,headers:{Allow:'GET, HEAD'}});
    // Only explicit published aliases can reach the private R2 bucket.
    const canonical = new URL('/_media-cache/'+key,url.origin);
    const cacheRequest = new Request(canonical,{method:request.method,headers:request.headers});
    const cached = await caches.default.match(cacheRequest);
    if (cached) return cached;
    const object = request.method === 'HEAD' ? await env.MEDIA.head(key) : await env.MEDIA.get(key,{onlyIf:request.headers,range:request.headers});
    if (!object) return new Response('Not found',{status:404});
    const headers = new Headers();
    object.writeHttpMetadata(headers);
    headers.set('ETag',object.httpEtag);
    headers.set('Accept-Ranges','bytes');
    headers.set('Cache-Control','public, max-age=3600');
    headers.set('X-Content-Type-Options','nosniff');
    if (request.headers.get('If-None-Match')?.split(',').map(x=>x.trim()).includes(object.httpEtag)) return new Response(null,{status:304,headers});
    if (request.method==='HEAD') { headers.set('Content-Length',String(object.size)); return new Response(null,{headers}); }
    if (!('body' in object)) return new Response(null,{status:412,headers});
    let status=200;
    if (object.range && request.headers.has('Range')) {
      const range=object.range;
      const length=range.length ?? Math.min(range.suffix ?? object.size,object.size);
      const offset=range.offset ?? object.size-length;
      headers.set('Content-Range',`bytes ${offset}-${offset+length-1}/${object.size}`);
      headers.set('Content-Length',String(length)); status=206;
    } else headers.set('Content-Length',String(object.size));
    const response=new Response(object.body,{status,headers});
    if(status===200) ctx.waitUntil(caches.default.put(new Request(canonical),response.clone()));
    return response;
  }
};
