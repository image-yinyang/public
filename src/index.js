import { nanoid } from 'nanoid';
import { Ai } from '@cloudflare/ai';
import OpenAI from 'openai';
import { Router, createCors, error, html } from 'itty-router';

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

async function imageAnalysisAndPrompts(requestId, request, env, ai, url) {
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
		return new Response(estr, { status: 500 });
	}

	const { content } = response.choices[0].message;
	if (!content) {
		return new Response(null, { status: 418 });
	}

	const threshold = Number.parseFloat(await env.ConfigKVStore.get('goodThreshold'));
	let thresholdMod;
	if (request.headers.has('X-Yinyang-Threshold-Mod')) {
		thresholdMod = Number.parseFloat(request.headers.get('X-Yinyang-Threshold-Mod'));
	}

	let sentences;
	try {
		sentences = await Promise.all(
			content
				.replaceAll(/\n/g, ' ')
				.replaceAll(/[^\w\d\.,\- ]/g, '')
				.split('. ')
				.map(async (sentence) => ({
					sentence,
					sentiment: await _text(ai, sentence, threshold, thresholdMod),
				})),
		);
	} catch (e) {
		const estr = `Sentiment failed: ${e}`;
		console.error(estr);
		return new Response(estr, { status: 500 });
	}

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
	return Response.json(responseObj);
}

async function fetchAndPersistInput(env, imageUrl) {
	if (new URL(imageUrl).origin === 'https://inputs.yinyang.computerpho.be') {
		return imageUrl;
	}

	const checkRes = await env.DB.prepare('select * from Inputs where SourceUrl = ?').bind(imageUrl).all();
	if (!checkRes.success) {
		console.error(`check fail: ${JSON.stringify(checkRes)}`);
		return;
	}

	if (checkRes.results.length) {
		const {
			results: [{ BucketId }],
		} = checkRes;
		console.log(`${imageUrl} is already persisted at ${BucketId}!`);
		return `https://inputs.yinyang.computerpho.be/${BucketId}`;
	}

	const resp = await fetch(imageUrl);
	if (!resp.ok) {
		console.error(`fetchAndPersistInput ${imageUrl}: ${resp.status}`);
		console.error(resp.message);
		return;
	}

	const cType = resp.headers.get('content-type');
	if (!cType) {
		console.error(`No content-type given for ${imageUrl}`);
		return;
	}

	const bucketId = `${nanoid()}.${cType.split('/').slice(-1)[0]}`;
	const { etag, size } = await env.INPUT_IMAGES_BUCKET.put(bucketId, await resp.blob());
	const insertRes = await env.DB.prepare('insert into Inputs values (?, ?, ?)').bind(imageUrl, cType, bucketId).run();

	if (!insertRes.success || !etag || !size) {
		console.error(`Bad persist ${insertRes.success} ${etag} ${size}`);
		return;
	}

	console.log(`Persisted input ${bucketId}, ${size} bytes, etag: ${etag}`);
	return `https://inputs.yinyang.computerpho.be/${bucketId}`;
}

async function checkAllowedHost(env, origin, request) {
	const allowedHosts = JSON.parse(await env.CommonKVStore.get('allowedHostsJSON'));
	if (!allowedHosts.includes(origin)) {
		warnWithCfInfo(request, `Disallowed origin: ${origin}`);
		return new Response(null, { status: 405 });
	}
}
async function getReqHandler(env, request) {
	const checkReqId = new URL(request.url).pathname?.slice(1);
	if (!checkReqId) {
		return new Response(null, { status: 404 });
	}

	// no need to deal with eventual consistency here: just let the client try again later!
	const reqObj = JSON.parse(await env.RequestsKVStore.get(checkReqId));

	if (!reqObj || reqObj.status === 'pending') {
		return new Response(null, { status: 202 });
	}

	return Response.json(reqObj);
}

async function postReqHandler(env, request) {
	const newReqId = nanoid(Number.parseInt(await env.ConfigKVStore.get('requestIdLengthBytes')));
	const body = await request.text();
	logWithCfInfo(request, `imageUrl: ${body}`);
	let ourUrl = await fetchAndPersistInput(env, body);
	if (!ourUrl) {
		console.error(`Bad fetch!! Falling back to original...`);
		ourUrl = body;
	}
	return imageAnalysisAndPrompts(newReqId, request, env, new Ai(env.AI), ourUrl);
}

export default {
	async fetch(request, env) {
		const origin = request.headers.get('origin');
		const { preflight, corsify } = createCors({
			methods: ['GET', 'POST'],
			origins: [origin],
		});

		const router = Router();

		router
			.all('*', checkAllowedHost.bind(null, env, origin))
			.all('*', preflight)
			.get('*', getReqHandler.bind(null, env))
			.post('/', postReqHandler.bind(null, env))
			.all('*', () => error(404));

		const ourError = (...args) => {
			console.error('Router middleware threw:');
			console.error(...args);
			return error(...args);
		};

		return router.handle(request).then(html).catch(ourError).then(corsify);
	},
};
