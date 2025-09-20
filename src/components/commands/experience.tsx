import type { ComponentCommand } from '@commands';
import { i18n } from '@locale';
import { useStore } from '@nanostores/preact';
import type { FunctionalComponent } from 'preact';

const messages = i18n('experience', {
	chairperson: 'Chairperson',
	chairpersonCompany: 'IEEE CIS PIET',
	chairpersonDescription:
		'Leading IEEE CIS Student Branch Chapter, organizing events and workshops, building industry partnerships, and implementing outreach initiatives for social impact.',
	chairpersonDuration: 'Dec 2023 - Present',
	chairpersonLocation: 'Jaipur, India',
	current: 'AI Research Analyst',
	currentCompany: 'BIZ4GROUP LLC',
	currentDescription:
		'Conducting extensive research in AI/ML, analyzing academic papers, exploring innovative algorithms, and collaborating with cross-functional teams for practical implementation.',
	currentDuration: 'June 2025 - Present',
	currentLocation: 'Jaipur, India',
	founder: 'Founder & CEO',
	founderCompany: 'PIE-STAR Interactive Studios',
	founderDescription:
		'Founded game and app development startup, led team in designing and launching mobile apps/games, managed strategic planning and operations.',
	founderDuration: 'Jan 2022 - Feb 2024',
	founderLocation: 'Jaipur, India',
	intern: 'AI/ML Intern',
	internCompany: 'Zeetron Networks Pvt. Ltd.',
	internDescription:
		'Built and deployed machine learning models using Python, gained experience in NLP, data visualization, and applied ML techniques to real-world datasets.',
	internDuration: 'Jul 2023 - Sep 2023',
	internLocation: 'Jaipur, India',
	robotics: 'Robotics Trainee',
	roboticsCompany: 'Rajasthan Centre of Advanced Technology',
	roboticsDescription:
		'Gained hands-on experience in robotics design, programming, sensor integration, and developed automation systems for industrial applications.',
	roboticsDuration: 'Apr 2024 - Sep 2024',
	roboticsLocation: 'Jaipur, India',
	title: 'Professional Experience',
	writer: 'Technical Content Writer',
	writerCompany: 'GeeksforGeeks',
	writerDescription:
		'Wrote tutorials and technical articles on programming concepts, simplified complex topics for students, ensured SEO optimization and editorial consistency.',
	writerDuration: 'Jan 2023 - Jul 2023',
	writerLocation: 'Jaipur, India',
});

const Experience: FunctionalComponent = () => {
	const t = useStore(messages);

	return (
		<div className='terminal-line-history'>
			<h3>{t.title}</h3>

			<div style={{ marginBottom: '1.5rem' }}>
				<p>
					<strong>{t.current}</strong>
				</p>
				<p>
					{t.currentCompany} | {t.currentDuration}
				</p>
				<p>{t.currentLocation}</p>
				<p style={{ color: 'var(--color-text-200)', marginTop: '0.5rem' }}>{t.currentDescription}</p>
			</div>

			<div style={{ marginBottom: '1.5rem' }}>
				<p>
					<strong>{t.chairperson}</strong>
				</p>
				<p>
					{t.chairpersonCompany} | {t.chairpersonDuration}
				</p>
				<p>{t.chairpersonLocation}</p>
				<p style={{ color: 'var(--color-text-200)', marginTop: '0.5rem' }}>{t.chairpersonDescription}</p>
			</div>

			<div style={{ marginBottom: '1.5rem' }}>
				<p>
					<strong>{t.robotics}</strong>
				</p>
				<p>
					{t.roboticsCompany} | {t.roboticsDuration}
				</p>
				<p>{t.roboticsLocation}</p>
				<p style={{ color: 'var(--color-text-200)', marginTop: '0.5rem' }}>{t.roboticsDescription}</p>
			</div>

			<div style={{ marginBottom: '1.5rem' }}>
				<p>
					<strong>{t.founder}</strong>
				</p>
				<p>
					{t.founderCompany} | {t.founderDuration}
				</p>
				<p>{t.founderLocation}</p>
				<p style={{ color: 'var(--color-text-200)', marginTop: '0.5rem' }}>{t.founderDescription}</p>
			</div>

			<div style={{ marginBottom: '1.5rem' }}>
				<p>
					<strong>{t.intern}</strong>
				</p>
				<p>
					{t.internCompany} | {t.internDuration}
				</p>
				<p>{t.internLocation}</p>
				<p style={{ color: 'var(--color-text-200)', marginTop: '0.5rem' }}>{t.internDescription}</p>
			</div>

			<div style={{ marginBottom: '1rem' }}>
				<p>
					<strong>{t.writer}</strong>
				</p>
				<p>
					{t.writerCompany} | {t.writerDuration}
				</p>
				<p>{t.writerLocation}</p>
				<p style={{ color: 'var(--color-text-200)', marginTop: '0.5rem' }}>{t.writerDescription}</p>
			</div>
		</div>
	);
};

const ExperienceCommand: ComponentCommand = {
	command: 'experience',
	component: Experience,
};

export default ExperienceCommand;
