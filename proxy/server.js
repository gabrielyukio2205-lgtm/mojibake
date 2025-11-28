 server.js - Proxy J.A.D.E. v3.0 (Universal)
 Agora suporta Chat, Code e Scholar automaticamente!

import express from express;
import fetch from node-fetch;
import cors from cors;

const app = express();
const PORT = process.env.PORT  10000;

 URL base do seu Hugging Face Space (Backend Real)
const HF_SPACE_URL = httpsmadras1-jade-port.hf.space;

app.use(cors());

 Aumentamos o limite para aceitar imagens grandes e áudios
app.use(express.json({ limit '50mb' }));  Aumentei pra 50MB pra garantir ;)

 Rota de saúde
app.get(health, (req, res) = {
  res.status(200).json({ status online, target HF_SPACE_URL });
});

 Rota Raiz (apenas visual)
app.get(, (req, res) = {
  res.send(`
    stylebody{font-familysans-serif;background#111;color#eee;displayflex;justify-contentcenter;align-itemscenter;height100vh;}style
    div style=text-aligncenter
      h1💎 Proxy J.A.D.E. Universalh1
      p style=color#4ade80Status Operacionalp
      pRedirecionando tráfego para b${HF_SPACE_URL}bp
    div
  `);
});

 ==================================================================
 🔄 O CORAÇÃO DO PROXY UNIVERSAL
 Captura qualquer método (GET, POST) em qualquer rota ()
 e repassa exatamente igual para o Hugging Face.
 ==================================================================
app.all(, async (req, res) = {
   Ignora a raiz e health check que já tratamos acima
  if (req.path ===   req.path === health) return;

  const targetUrl = HF_SPACE_URL + req.path;
  const method = req.method;

  console.log(`➡️ [Proxy] Encaminhando ${method} ${req.path} - ${targetUrl}`);

  try {
    const headers = {
      Content-Type applicationjson
    };

     Se você tiver token privado no futuro, descomente abaixo
     if (process.env.HF_TOKEN) headers[Authorization] = `Bearer ${process.env.HF_TOKEN}`;

     Configuração da requisição para o HF
    const fetchOptions = {
      method method,
      headers headers,
    };

     Só anexa o corpo se não for GET ou HEAD (GET com corpo dá erro)
    if (method !== GET && method !== HEAD) {
      fetchOptions.body = JSON.stringify(req.body);
    }

     Faz a chamada real
    const hfResponse = await fetch(targetUrl, fetchOptions);

     Tenta pegar o JSON da resposta
     Se o backend devolver áudio binário ou arquivo, precisamos tratar diferente,
     mas por enquanto, como seu app.py devolve JSON (com base64 dentro), isso funciona.
    const data = await hfResponse.json();

    console.log(`⬅️ [Proxy] Resposta recebida do Space ${hfResponse.status}`);
    res.status(hfResponse.status).json(data);

  } catch (err) {
    console.error(`❌ [Proxy Erro]`, err);
    res.status(500).json({
      success false,
      error Erro de comunicação no Proxy.,
      details err.message
    });
  }
});

app.listen(PORT, () = {
  console.log(`🚀 Proxy J.A.D.E. Universal rodando na porta ${PORT}`);
});