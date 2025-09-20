import type { ComponentCommand } from '@commands';
import { i18n } from '@locale';
import { useStore } from '@nanostores/preact';
import type { FunctionalComponent } from 'preact';

const messages = i18n('skills', {
	ai: 'AI & Machine Learning',
	aiList: 'TensorFlow, PyTorch, Scikit-learn, OpenAI API, Computer Vision',
	aiTools: 'AI Tools & Platforms',
	aiToolsList: 'Cursor, Windsurf, Firebase Studio, ChatGPT, Claude, GitHub Copilot, Replit',
	backend: 'Backend Technologies',
	backendList: 'Node.js, Express.js, FastAPI, Django, REST APIs, GraphQL',
	database: 'Databases',
	databaseList: 'MongoDB, PostgreSQL, MySQL, Redis, Firebase',
	frontend: 'Frontend Technologies',
	frontendList: 'React, Vue.js, HTML5, CSS3, Tailwind CSS, Astro',
	programming: 'Programming Languages',
	programmingList: 'JavaScript, TypeScript, Python, Java, C++, SQL',
	promptEngineering: 'Prompt Engineering',
	promptEngineeringList: 'ChatGPT Prompting, Claude Optimization, AI Model Fine-tuning, Context Engineering',
	title: 'Technical Skills',
	tools: 'Tools & Technologies',
	toolsList: 'Git, Docker, AWS, Linux, VS Code, Figma, Postman',
});

const Skills: FunctionalComponent = () => {
	const t = useStore(messages);

	return (
		<div className='terminal-line-history'>
			<h3>{t.title}</h3>

			<div style={{ marginBottom: '1rem' }}>
				<p>
					<strong>{t.programming}</strong>
				</p>
				<p style={{ color: 'var(--color-text-200)', marginLeft: '1rem' }}>{t.programmingList}</p>
			</div>

			<div style={{ marginBottom: '1rem' }}>
				<p>
					<strong>{t.frontend}</strong>
				</p>
				<p style={{ color: 'var(--color-text-200)', marginLeft: '1rem' }}>{t.frontendList}</p>
			</div>

			<div style={{ marginBottom: '1rem' }}>
				<p>
					<strong>{t.backend}</strong>
				</p>
				<p style={{ color: 'var(--color-text-200)', marginLeft: '1rem' }}>{t.backendList}</p>
			</div>

			<div style={{ marginBottom: '1rem' }}>
				<p>
					<strong>{t.database}</strong>
				</p>
				<p style={{ color: 'var(--color-text-200)', marginLeft: '1rem' }}>{t.databaseList}</p>
			</div>

			<div style={{ marginBottom: '1rem' }}>
				<p>
					<strong>{t.ai}</strong>
				</p>
				<p style={{ color: 'var(--color-text-200)', marginLeft: '1rem' }}>{t.aiList}</p>
			</div>

			<div style={{ marginBottom: '1rem' }}>
				<p>
					<strong>{t.aiTools}</strong>
				</p>
				<p style={{ color: 'var(--color-text-200)', marginLeft: '1rem' }}>{t.aiToolsList}</p>
			</div>

			<div style={{ marginBottom: '1rem' }}>
				<p>
					<strong>{t.promptEngineering}</strong>
				</p>
				<p style={{ color: 'var(--color-text-200)', marginLeft: '1rem' }}>{t.promptEngineeringList}</p>
			</div>

			<div style={{ marginBottom: '1rem' }}>
				<p>
					<strong>{t.tools}</strong>
				</p>
				<p style={{ color: 'var(--color-text-200)', marginLeft: '1rem' }}>{t.toolsList}</p>
			</div>
		</div>
	);
};

const SkillsCommand: ComponentCommand = {
	command: 'skills',
	component: Skills,
};

export default SkillsCommand;
