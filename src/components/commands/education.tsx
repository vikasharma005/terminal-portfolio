import type { ComponentCommand } from '@commands';
import { i18n } from '@locale';
import { useStore } from '@nanostores/preact';
import type { FunctionalComponent } from 'preact';

const messages = i18n('education', {
	degree: 'Bachelor of Technology in Artificial Intelligence and Data Science',
	description:
		'Specialized in AI, Machine Learning, NLP, Data Mining, Big Data Analytics, Computer Vision, and Cloud Computing.',
	highSchool: 'Secondary Examination',
	highSchoolDescription: 'CBSE curriculum with Mathematics, Science, Social Studies, and Languages.',
	highSchoolLocation: 'Alwar, India',
	highSchoolName: 'National Academy English Medium Sen Sec School',
	highSchoolWebsite: 'www.nationalacademyedu.com',
	highSchoolYear: 'Jul 2018 - Jun 2019',
	location: 'Jaipur, India',
	secondary: 'Senior Secondary Examination (Mathematics and Computer Science)',
	secondaryDescription: 'CBSE curriculum with focus on Mathematics, Computer Science, Physics, and English.',
	secondaryLocation: 'Jaipur, India',
	secondarySchool: 'Tagore International School (CBSE)',
	secondaryWebsite: 'www.tagoreint.com',
	secondaryYear: 'Aug 2020 - Aug 2021',
	title: 'Education',
	university: 'Poornima Institute of Engineering & Technology',
	website: 'www.poornima.org',
	year: 'Nov 2021 - May 2025',
});

const Education: FunctionalComponent = () => {
	const t = useStore(messages);

	return (
		<div className='terminal-line-history'>
			<h3>{t.title}</h3>

			<div style={{ marginBottom: '1.5rem' }}>
				<p>
					<strong>{t.degree}</strong>
				</p>
				<p>{t.university}</p>
				<p>
					{t.year} | {t.location}
				</p>
				<p style={{ color: 'var(--color-primary)', fontSize: '0.9rem' }}>{t.website}</p>
				<p style={{ color: 'var(--color-text-200)', marginTop: '0.5rem' }}>{t.description}</p>
			</div>

			<div style={{ marginBottom: '1.5rem' }}>
				<p>
					<strong>{t.secondary}</strong>
				</p>
				<p>{t.secondarySchool}</p>
				<p>
					{t.secondaryYear} | {t.secondaryLocation}
				</p>
				<p style={{ color: 'var(--color-primary)', fontSize: '0.9rem' }}>{t.secondaryWebsite}</p>
				<p style={{ color: 'var(--color-text-200)', marginTop: '0.5rem' }}>{t.secondaryDescription}</p>
			</div>

			<div style={{ marginBottom: '1rem' }}>
				<p>
					<strong>{t.highSchool}</strong>
				</p>
				<p>{t.highSchoolName}</p>
				<p>
					{t.highSchoolYear} | {t.highSchoolLocation}
				</p>
				<p style={{ color: 'var(--color-primary)', fontSize: '0.9rem' }}>{t.highSchoolWebsite}</p>
				<p style={{ color: 'var(--color-text-200)', marginTop: '0.5rem' }}>{t.highSchoolDescription}</p>
			</div>
		</div>
	);
};

const EducationCommand: ComponentCommand = {
	command: 'education',
	component: Education,
};

export default EducationCommand;
