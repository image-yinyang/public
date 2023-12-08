import { nanoid } from 'nanoid';
import { Ai } from '@cloudflare/ai';
import OpenAI from 'openai';

function _logWithCfInfo(method, request, ...args) {
	const { city, region, continent, asOrganization } = request.cf;
	return console[method](`[${request.headers.get('x-real-ip')} / ${city}, ${region}, ${continent} / ${asOrganization}]`, ...args);
}

const logWithCfInfo = _logWithCfInfo.bind(null, 'log');
const warnWithCfInfo = _logWithCfInfo.bind(null, 'warn');

async function _text(ai, userText, threshold, thresholdMod) {
	const inputs = { text: userText };
	let response = await ai.run('@cf/huggingface/distilbert-sst-2-int8', inputs);

	if (thresholdMod) {
		threshold += thresholdMod / 10.0;
	}

	const negative = response.find(({ label }) => label === 'NEGATIVE').score;
	const positive = response.find(({ label }) => label === 'POSITIVE').score;
	return {
		negative,
		positive,
		good: positive - negative > threshold,
	};
}

function _promptFromSentences(sentences, condFunc) {
	return sentences
		.filter(({ sentiment: { good } }) => condFunc(good))
		.map(({ sentence }) => sentence)
		.join('. ');
}

async function imageAnalysisAndPrompts(requestId, request, env, ai, headers, url) {
	let openaiKey = request.headers.get('X-Yinyang-OpenAI-Key');
	const allowBuiltIns = JSON.parse(env.USE_BUILTIN_OPENAI_KEY);
	if (allowBuiltIns.includes(openaiKey)) {
		logWithCfInfo(request, `Request using builtin OpenAI key!`);
		openaiKey = env.OPENAI_KEY;
	}

	if (!openaiKey) {
		return new Response(null, { status: 401 });
	}

	const openai = new OpenAI({
		baseURL: 'https://gateway.ai.cloudflare.com/v1/2878d2dc2b7536339d881d884d27d775/yinyang-open-ai/openai',
		apiKey: openaiKey,
	});
	const prompt = await env.ConfigKVStore.get('prompt');
	const detailLevel = await env.ConfigKVStore.get('detail');
	let response;

	try {
		response = await openai.chat.completions.create({
			model: 'gpt-4-vision-preview',
			max_tokens: 4096,
			messages: [
				{
					role: 'user',
					content: [
						{ type: 'text', text: prompt },
						{
							type: 'image_url',
							image_url: {
								url,
								detail: detailLevel,
							},
						},
					],
				},
			],
		});
	} catch (e) {
		const estr = `OpenAI failed: ${e}`;
		console.error(estr);
		return new Response(estr, { headers, status: 500 });
	}

	const { content } = response.choices[0].message;
	if (!content) {
		return new Response(null, { headers, status: 418 });
	}

	const threshold = Number.parseFloat(await env.ConfigKVStore.get('goodThreshold'));
	let thresholdMod;
	if (request.headers.has('X-Yinyang-Threshold-Mod')) {
		thresholdMod = Number.parseFloat(request.headers.get('X-Yinyang-Threshold-Mod'));
	}

	const sentences = await Promise.all(
		content
			.replaceAll(/\n/g, ' ')
			.replaceAll(/[^\w\d\.,\- ]/g, '')
			.split('. ')
			.map(async (sentence) => ({
				sentence,
				sentiment: await _text(ai, sentence, threshold, thresholdMod),
			})),
	);

	const goodPrompt = _promptFromSentences(sentences, (good) => good);
	const badPrompt = _promptFromSentences(sentences, (good) => !good);

	const results = {
		good: {
			prompt: goodPrompt,
			imageBucketId: null,
		},
		bad: {
			prompt: badPrompt,
			imageBucketId: null,
		},
	};

	const {
		model,
		usage: { total_tokens },
	} = response;

	const responseObj = {
		input: {
			url,
			threshold,
			thresholdMod,
		},
		createdTimeUnixMs: +new Date(),
		requestorIp: request.headers.get('x-real-ip'),
		requestId,
		response: content,
		sentences,
		results,
		meta: {
			openai_tokens_used: total_tokens,
			openai_full_model_used: model,
			openai_prompt: prompt,
		},
	};

	await env.RequestsKVStore.put(
		requestId,
		JSON.stringify({
			...responseObj,
			status: 'pending',
		}),
	);

	await env.GENIMG_REQ_QUEUE.send({ requestId });

	logWithCfInfo(request, `Posted image generation request for ${requestId}`);
	return Response.json(responseObj, { headers });
}

export default {
	async fetch(request, env) {
		const allowedHosts = JSON.parse(await env.CommonKVStore.get('allowedHostsJSON'));
		const origin = request.headers.get('origin');
		if (!allowedHosts.includes(origin)) {
			warnWithCfInfo(request, `Disallowed origin: ${origin}`);
			return new Response(null, { status: 405 });
		}

		const headers = new Headers();
		headers.set('Access-Control-Allow-Origin', origin);

		if (!['GET', 'POST', 'OPTIONS'].includes(request.method)) {
			warnWithCfInfo(request, `Bad method: ${request.method} ${request.url}`);
			return new Response(null, { headers, status: 405 });
		}

		if (request.method === 'OPTIONS') {
			headers.set('Access-Control-Allow-Headers', 'X-Yinyang-OpenAI-Key,X-Yinyang-Threshold-Mod');
			return new Response(null, { status: 200, headers });
		} else if (request.method === 'GET') {
			const checkReqId = new URL(request.url).pathname?.slice(1);
			if (!checkReqId) {
				return new Response(null, { headers, status: 404 });
			}

			// no need to deal with eventual consistency here: just let the client try again later!
			const reqObj = JSON.parse(await env.RequestsKVStore.get(checkReqId));

			if (!reqObj || reqObj.status === 'pending') {
				return new Response(null, { headers, status: 202 });
			}

			return Response.json(reqObj, { headers });
		} else if (request.method === 'POST') {
			const newReqId = nanoid(Number.parseInt(await env.ConfigKVStore.get('requestIdLengthBytes')));
			const body = await request.text();
			logWithCfInfo(request, `imageUrl: ${body}`);
			return imageAnalysisAndPrompts(newReqId, request, env, new Ai(env.AI), headers, body);
		}
	},
};
